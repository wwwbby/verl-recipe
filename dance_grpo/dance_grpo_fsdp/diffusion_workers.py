# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import datetime
import json
import logging
import math
import os
import re
import tempfile
import time

import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
from hpsv3 import HPSv3RewardInferencer
from omegaconf import DictConfig
from pandas.core.dtypes.cast import LossySetitemError
from PIL import Image
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType

from recipe.dance_grpo.actor import DataParallelPPOActor
from recipe.dance_grpo.rollout import HFRollout
from recipe.dance_grpo.utils import init_fsdp_module
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import (
    Dispatch, make_nd_compute_dataproto_dispatch_fn, register)
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import (get_device_id, get_device_name,
                               get_nccl_backend, get_torch_device)
from verl.utils.fsdp_utils import (load_fsdp_model_to_gpu, load_fsdp_optimizer,
                                   offload_fsdp2_model_to_cpu,
                                   offload_fsdp_model_to_cpu,
                                   offload_fsdp_optimizer)
from verl.utils.profiler import (DistProfiler, DistProfilerExtension,
                                 ProfilerConfig, log_gpu_memory_usage,
                                 simple_timer)
from verl.workers.sharding_manager.fsdp_ulysses import \
    FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

device_name = get_device_name()

from typing import Tuple

import torch.nn as nn
from qwen_vl_utils import process_vision_info
from torch.distributed._composable.fsdp import CPUOffloadPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, checkpoint_wrapper)
from torch.distributed.fsdp import fully_shard
from transformers import AutoProcessor


def create_device_mesh(world_size, fsdp_size):
    """Create device mesh for FSDP"""
    if fsdp_size <= 0 or fsdp_size > world_size:
        fsdp_size = world_size
    return init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["dp", "fsdp"])


class DiffusionActorRolloutWorker(Worker, DistProfilerExtension):
    """
    Worker for diffusion action rollout and GRPO training
    This worker encapsulates:
    1. Rollout process with diffusion model sampling and log probability calculation
    2. Reward calculation logic using GRPO
    3. GRPO policy update with advantage clipping and KL divergence regularization
    4. Checkpointing and loading of model and optimizer states
    """

    def __init__(self, config: DictConfig, role='hybrid', **kwargs):
        log_gpu_memory_usage("Before Diffusion Worker init", logger=logger, level=logging.INFO)
        Worker.__init__(self)

        self.config = config
        self.role = role
        self._is_actor = role in ["actor", "hybrid", "actor_rollout"]
        self._is_rollout = role in ["rollout", "hybrid"]

        # Initialize distributed training
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Build device mesh for FSDP
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # Build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # Create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=rank, is_collect=True)
     
        self._register_dispatch_collect_info("rollout", dp_rank=rank, is_collect=True)

        # 只有当序列并行大小大于1时才创建分片管理器
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        else:
            self.ulysses_sharding_manager = None

        # Initialize models
        self.actor_module = None
        self.actor_module_fsdp = None
        self.processor = None
        self.ref_module = None
        self.ref_module_fsdp = None
        self.full_model = None
        self.dit = None

        # Initialize optimizers and schedulers
        self.actor_optimizer = None
        self.actor_lr_scheduler = None
      
        # Checkpoint management
        self.checkpoint_manager = None
        self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
        self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
      
        # Setup FSDP
        self._build_model_optimizer()
        log_gpu_memory_usage("After Diffusion Worker init", logger=logger, level=logging.INFO)

    def apply_fsdp2_and_ac(
        self,
        model: nn.Module, 
        no_split_module_classes: list[str], 
        is_train: bool = False,
        cpu_offload: bool = False
    ) -> nn.Module:
        """
        Applies FSDP2 fully_shard, CPU offloading, and Activation Checkpointing
        to sub-modules based on their class names.
        """
        from torch.distributed.fsdp import MixedPrecisionPolicy
        offload_policy = CPUOffloadPolicy()

        # param_dtype=bf16: forward/backward compute in bf16 (memory efficient).
        # The actual stored (sharded) parameters stay in their original dtype
        # (fp32 thanks to dit_model.float() before wrapping), so the optimizer
        # operates on fp32 master weights.  reduce_dtype=fp32 ensures gradient
        # all-reduce preserves full precision.
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ) if is_train else None
  
        # Iterate over modules to wrap specific layer blocks (e.g. Decoder layers, Vision blocks)
        for name, module in model.named_modules():
            if module.__class__.__name__ in no_split_module_classes:
                if is_train:
                    # Apply gradient checkpointing only for trainable modules
                    checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
                # Apply FSDP2 fully_shard with CPU offloading
                shard_kwargs = {}
                if cpu_offload:
                    shard_kwargs['offload_policy'] = offload_policy
                if mp_policy is not None:
                    shard_kwargs['mp_policy'] = mp_policy
                fully_shard(module, **shard_kwargs)
          
        # Finally, wrap the root module
        shard_kwargs = {}
        if cpu_offload:
            shard_kwargs['offload_policy'] = offload_policy
        if mp_policy is not None:
            shard_kwargs['mp_policy'] = mp_policy
        fully_shard(model, **shard_kwargs)
        return model

    def extract_and_wrap_models(
        self,
        model_id_or_path: str,
    ):
        """
        Extracts the MLLM, DiT, and VAE components from recipe.dance_grpo.mammothmoda2Model.
        Wraps MLLM and DiT with FSDP2, with DiT set for training and others for eval.
        """
        if torch.distributed.get_rank() == 0:
            logger.info(f"Loading Mammothmoda2Model from {model_id_or_path}...")

        # Import from the local mammothmoda2 package
        from recipe.dance_grpo.mammothmoda2.model import (DEFAULT_NEGATIVE_PROMPT,
                                        Mammothmoda2Model)

        # Load base model (weights kept on CPU initially to save memory before FSDP wrapping)
        full_model = Mammothmoda2Model.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.bfloat16,
            t2i_generate=True,
        ).to(torch.bfloat16)
        processor = AutoProcessor.from_pretrained(
            model_id_or_path, 
            t2i_generate=True,
            ar_height=32,
            ar_width=32,
        )

        # 1. Extract the modules
        mllm_model = full_model.llm_model       # Qwen3-VL based text encoder
        dit_model = full_model.gen_transformer  # Transformer2DModel (DiT)
        vae_model = full_model.gen_vae          # AutoencoderKL

        # 2. Configure train/eval modes & requires_grad
        # The training only happens on DiT, so all other components are set to eval and no_grad
        mllm_model.eval()
        for param in mllm_model.parameters():
            param.requires_grad = False

        vae_model.eval()
        for param in vae_model.parameters():
            param.requires_grad = False
        vae_model.to('npu')

        dit_model.train()
        
        # Determine parameter freezing based on configuration
        freeze_non_attn_ffn = getattr(self.config.actor, "freeze_non_attn_ffn", False)
        
        if freeze_non_attn_ffn:
            # Freeze all DiT parameters first, then selectively enable attention and FFN layers.
            for param in dit_model.parameters():
                param.requires_grad = False
            _attn_ffn_re = re.compile(
                r'\battn\b|\battention\b|self_attn|cross_attn|\.ff\.|'
                r'\.ff1\.|ff\.net|\bmlp\b|\.ffn\.|feed_forward'
            )
            for name, param in dit_model.named_parameters():
                if _attn_ffn_re.search(name):
                    param.requires_grad = True
        else:
            # Full fine-tuning: unfreeze all parameters in DiT
            for param in dit_model.parameters():
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in dit_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in dit_model.parameters())
        if torch.distributed.get_rank() == 0:
            logger.info(
                f"DiT trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100.0 * trainable_params / max(total_params, 1):.1f}%) "
                f"[{'attention + FFN layers only' if freeze_non_attn_ffn else 'full fine-tuning'}]"
            )

        # 3. Retrieve module boundaries for FSDP wrapping defined by the model
        mllm_no_split = getattr(mllm_model, "_no_split_modules", [])
        dit_no_split = getattr(dit_model, "_no_split_modules", [])

        # 4. Wrap with FSDP2
        # Convert DiT to fp32 BEFORE FSDP2 wrapping so that the optimizer
        # operates on fp32 master weights.  FSDP2's param_dtype=bf16 handles
        # bf16 compute during forward/backward, but the stored (sharded)
        # parameters stay fp32.  Without this, tiny Adam updates (~lr ≈ 1e-6)
        # are rounded to zero in bf16 (min step ≈ 0.008 near value 1.0),
        # causing the model to never learn.
        dit_model = dit_model.float()
        if torch.distributed.get_rank() == 0:
            logger.info("Applying FSDP2 fully_shard, gradient checkpointing, and CPU offloading...")
            logger.info(f"DiT param dtype before FSDP2: {next(dit_model.parameters()).dtype}")
        mllm_model = self.apply_fsdp2_and_ac(mllm_model, mllm_no_split, is_train=False, cpu_offload=True)
        dit_model = self.apply_fsdp2_and_ac(dit_model, dit_no_split, is_train=True, cpu_offload=True)
        dit_model.enable_gradient_checkpointing()

        # VAE is kept small and frozen, you may map it directly to the device when needed

        return mllm_model, dit_model, vae_model, full_model, processor

    def _build_model_optimizer(self):
        """Setup FSDP for distributed training"""
        # Apply monkey patches
        # Setup actor model with FSDP
        log_gpu_memory_usage("Before init_fsdp_module", logger=logger, level=logging.INFO)

        mllm, dit, vae, full_model, processor = self.extract_and_wrap_models(self.config.model.path)

        if torch.distributed.get_rank() == 0:
            logger.info("Successfully loaded model.")

        self.dit = dit
        self.full_model = full_model
        self.processor = processor

        # Build optimizer using AdamW with parameters from train_grpo_edit.py
        # Collect all trainable parameters from the model
        params_to_optimize = []
        params_to_optimize.extend(list(filter(lambda p: p.requires_grad, self.dit.parameters())))

        self.actor_optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.config.actor.optim.lr,
            betas=(0.9, 0.999),
            weight_decay=self.config.actor.optim.weight_decay,
            eps=1e-8,
        )

        # Build LR scheduler using get_scheduler from diffusers.optimization
        from diffusers.optimization import get_scheduler
        self.actor_lr_scheduler = get_scheduler(
            self.config.actor.optim.lr_scheduler.name,
            optimizer=self.actor_optimizer,
            num_warmup_steps=self.config.actor.optim.lr_scheduler.num_warmup_steps,
            num_training_steps=self.config.actor.optim.lr_scheduler.num_training_steps,
            num_cycles=self.config.actor.optim.lr_scheduler.get("num_cycles", 1),
            power=self.config.actor.optim.lr_scheduler.get("power", 1.0),
        )

        # Create a custom config for HFRollout that includes diffusion-specific parameters
        rollout_config = self.config.rollout
        actor_config = self.config.actor
        self.rollout = HFRollout(mllm, dit, vae, full_model, self.processor, config=rollout_config)
        self.actor = DataParallelPPOActor(self.dit, actor_config)
        self.inferencer = HPSv3RewardInferencer(device='cpu', checkpoint_path=self.config.model.reward_model_path)
        self.inferencer.model.eval()
        for param in self.inferencer.model.parameters():
            param.requires_grad = False

        # Initialize checkpoint manager
        # if self._is_actor:
        #     checkpoint_contents = {
        #         # "load_contents": ["model", "optimizer", "lr_scheduler"],
        #         # "save_contents": ["model", "optimizer", "lr_scheduler"],
        #         "load_contents": ["model"],
        #         "save_contents": ["model"]
        #     }
        #     self.checkpoint_manager = FSDPCheckpointManager(
        #         model=self.actor_module_fsdp,
        #         optimizer=self.actor_optimizer,
        #         lr_scheduler=self.actor_lr_scheduler,
        #         processing_class=None,  # Diffusion worker doesn't use processor or tokenizer
        #         checkpoint_config=checkpoint_contents,
        #     )
        log_gpu_memory_usage("After init_fsdp_module", logger=logger, level=logging.INFO)

    def read_inputs_and_generate_latents(self, data: DataProto):
        """Read inputs from data and generate latents using mllm model"""
        # Get text and image/video inputs from data
        messages = data.non_tensor_batch['messages']
      
        if messages is None:
            raise ValueError("Messages are required for generate_latents")
        load_fsdp_model_to_gpu(self.actor_module_fsdp.text_encoder)
        # Preprocess inputs and generate latents using mllm model
        with torch.no_grad():
            # Extract prompt from messages
            message = messages[0]
            prompt = message[0]['content'][0]['text']
            negative_prompt = "" # messages[0]['content'][0]['negative_prompt']
          
            # Preprocess inputs
            text_embd_dict = self.actor_module_fsdp.preprocess(prompt, negative_prompt, None, None)
          
            # Generate latents and text embeddings
            z, text_hidden_states, negative_text_hidden_states = self.actor_module_fsdp.mllm_generate(
                text_embd_dict, None, None, None
            )
          
            # Create dummy masks (mammothmoda25 doesn't use these in the same way)
            batch_size = z.shape[0]
            seq_len = text_hidden_states.shape[1]
            text_attention_mask = torch.ones(batch_size, seq_len).to(z.device)
            negative_text_attention_mask = torch.ones(batch_size, seq_len).to(z.device)
          
            ref_latent = [None] * batch_size
          
            output = DataProto.from_dict(
                tensors={
                    'z': z,
                    'text_hidden_states': text_hidden_states,
                    'text_attention_mask': text_attention_mask,
                    'negative_text_hidden_states': negative_text_hidden_states,
                    'negative_text_attention_mask': negative_text_attention_mask,
                },
                non_tensors={
                    'ref_latents': np.array(ref_latent, dtype=object)
                },
            )
      
        # Update data with generated latents, embeddings
        data.batch = output.batch
        return data.union(output)

    def compute_rewards(self, data: DataProto):
        """Compute rewards for the generated sequences"""
        batch_size = data.batch.batch_size[0]
        scores_list = []

        prompts = data.non_tensor_batch['prompts'].tolist()
        all_images = data.non_tensor_batch['all_images']
        all_images_pil = [Image.fromarray(img_np) for img_np in all_images]

        def get_temp_paths(images):
            temp_paths = []

            for img in images:
                fd, path = tempfile.mkstemp(suffix='.png')
              
                try:
                    with os.fdopen(fd, 'wb') as tmp:
                        img.save(tmp, format='PNG')
                    temp_paths.append(path)
                except Exception as e:
                    os.close(fd)
                    logger.error(f"Error saving image: {e}")
                  
            return temp_paths

        def cleanup_temp_files(paths):
            for path in paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError as e:
                    logger.warning(f"Error deleting {path}: {e}")

        # Execute
        file_paths = get_temp_paths(all_images_pil)

        if self.inferencer.device == 'cpu':
            self.inferencer.model.to('npu')
            self.inferencer.device = 'npu'
        with torch.no_grad():
            rewards = self.inferencer.reward(file_paths, prompts)
        self.inferencer.model.to('cpu')
        self.inferencer.device = 'cpu'

        cleanup_temp_files(file_paths)

        scores = [reward[0].item() * 0.1 for reward in rewards]
        data.batch["rewards"] = torch.tensor(scores)

        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, data: DataProto):
        log_gpu_memory_usage("Before generate_sequences", logger=logger, level=logging.INFO)
        """Generate sequences using diffusion model with asynchronous processing"""
        # Step 1: Read inputs and generate latents using mllm model

        # Propagate step/experiment info to the rollout object so _generate_minibatch
        # can create the correct sub-directory structure.
        self.rollout._current_global_step = data.meta_info.get("global_step", 0)
        self.rollout._current_experiment_name = data.meta_info.get("experiment_name", "dance_grpo")

        # data = self.read_inputs_and_generate_latents(data)
        data = data.repeat(repeat_times=self.config.rollout.n, interleave=True)
      
        # Handle mammothmoda25 pipeline directly
        with torch.no_grad():
            # Get inputs from data
            # Generate videos using mammothmoda25's dit_generate
            output = self.rollout.generate_sequences(data)
        data = data.union(output)
        data = self.compute_rewards(data)

        # Persist per-rank JSONL metadata (prompt + reward + image path) for every rollout.
        self._save_rollout_metadata(data)

        log_gpu_memory_usage("After generate_sequences", logger=logger, level=logging.INFO)
        return data

    def _save_rollout_metadata(self, data: DataProto) -> None:
        """Write a JSONL file per rank with per-sample metadata (prompt, reward, image path)."""
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            global_step = data.meta_info.get("global_step", 0)
            experiment_name = data.meta_info.get("experiment_name", "dance_grpo")

            output_dir = str(getattr(self.config.rollout, "output_dir", "logs"))
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(os.getcwd(), output_dir)

            step_dir = os.path.join(output_dir, experiment_name, f"step_{global_step:06d}")
            os.makedirs(step_dir, exist_ok=True)

            prompts     = list(data.non_tensor_batch.get("prompts", []))
            image_paths = list(data.non_tensor_batch.get("image_paths", []))
            rewards     = data.batch.get("rewards", None)

            lines = []
            n_samples = len(prompts)
            for i in range(n_samples):
                entry = {
                    "index":       i,
                    "rank":        rank,
                    "global_step": global_step,
                    "prompt":      str(prompts[i]) if i < len(prompts) else "",
                    "reward":      float(rewards[i].item()) if rewards is not None and i < len(rewards) else None,
                    "image_path":  str(image_paths[i]) if i < len(image_paths) else "",
                }
                lines.append(json.dumps(entry))

            meta_path = os.path.join(step_dir, f"rank_{rank}_metadata.jsonl")
            with open(meta_path, "w") as f:
                f.write("\n".join(lines))
            logger.info(f"Saved rollout metadata ({n_samples} samples) -> {meta_path}")
        except Exception as exc:
            logger.warning(f"_save_rollout_metadata failed: {exc}")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        """Update the DiT actor with the DanceGRPO objective.

        Input ``data`` (produced by generate_sequences) is expected to contain:
          batch tensors:
            all_latents          – [B, T+1, C, H, W]  noisy latents at every SDE step
            all_log_probs        – [B, T]              log-probs recorded during rollout
            rewards              – [B]                 per-sample scalar rewards
            text_hidden_states   – [B, L, D]
            text_attention_mask  – [B, L]
            (optional) negative_text_hidden_states / negative_text_attention_mask
            (optional) image_hidden_states / image_attention_mask
          non-tensor:
            sigma_schedule       – numpy array [T+1]  time-steps used during rollout

        The MLLM (AR part) stays frozen; only the DiT parameters receive gradients.
        """
        import random
        from contextlib import nullcontext

        from recipe.dance_grpo.mammothmoda2.model.mammothmoda2_dit import RotaryPosEmbedReal

        log_gpu_memory_usage("Before update_actor", logger=logger)
        assert self._is_actor

        context_manager = self.ulysses_sharding_manager if self.ulysses_sharding_manager is not None else nullcontext()

        with context_manager:
            # -------------------------------------------------------------- #
            # 1.  Unpack rollout data                                          #
            # -------------------------------------------------------------- #
            data = data.to(get_device_name())
            all_latents    = data.batch["all_latents"].float()  # [B, T+1, C, H, W]  ensure float32
            old_log_probs  = data.batch["all_log_probs"].float()  # [B, T]  ensure float32
            rewards        = data.batch["rewards"].float()   # [B]  (already scaled at reward model output)

            text_hidden_states  = data.batch["text_hidden_states"]   # [B, L_padded, D]
            text_attention_mask = data.batch["text_attention_mask"]  # [B, L_padded]

            has_neg = "negative_text_hidden_states" in data.batch.keys()
            neg_text_hidden_states  = data.batch.get("negative_text_hidden_states", None)
            neg_text_attention_mask = data.batch.get("negative_text_attention_mask", None)

            has_img_cond = "image_hidden_states" in data.batch.keys()
            img_hidden_states  = data.batch.get("image_hidden_states", None)
            img_attention_mask = data.batch.get("image_attention_mask", None)

            # Actual (unpadded) sequence lengths stored by rollout for each conditioning tensor.
            # After DataProto.concat, different samples may have different original lengths.
            # Use the max across the batch so no sample's tokens get cut.
            text_actual_len = int(data.batch["text_seq_len"].max().item())
            neg_actual_len  = int(data.batch["neg_seq_len"].max().item()) if "neg_seq_len" in data.batch.keys() else None
            img_actual_len  = int(data.batch["image_seq_len"].max().item()) if "image_seq_len" in data.batch.keys() else None

            # Truncate padded conditioning back to original length so training DiT
            # sees the same sequence length that the rollout DiT used.
            # _pad_or_truncate_conditioning left-pads, so we keep the RIGHTMOST tokens.
            text_hidden_states  = text_hidden_states[:, -text_actual_len:, :]
            text_attention_mask = text_attention_mask[:, -text_actual_len:]
            if has_neg and neg_actual_len is not None:
                neg_text_hidden_states  = neg_text_hidden_states[:, -neg_actual_len:, :]
                neg_text_attention_mask = neg_text_attention_mask[:, -neg_actual_len:]
            if has_img_cond and img_actual_len is not None:
                img_hidden_states  = img_hidden_states[:, -img_actual_len:, :]
                img_attention_mask = img_attention_mask[:, -img_actual_len:]

            device = all_latents.device

            # -------------------------------------------------------------- #
            # 2.  Configuration                                                #
            # -------------------------------------------------------------- #
            actor_config   = self.config.actor
            clip_range     = actor_config.ppo_clip_range
            adv_clip_max   = actor_config.ppo_adv_clip_max
            max_grad_norm  = actor_config.ppo_max_grad_norm
            kl_coeff       = actor_config.ppo_kl_coeff
            micro_bs       = actor_config.micro_batch_size   # microbatch size for gradient accumulation
            grpo_size      = self.config.rollout.n           # samples per prompt group
            # train_cfg_scale > 1.0 enables classifier-free guidance during the training
            # forward pass (mirrors rollout cfg_scale).  Requires negative embeddings in data.
            train_cfg_scale = float(actor_config.get("train_cfg_scale", 1.0))

            # -------------------------------------------------------------- #
            # 3.  Sigma schedule & timestep selection                         #
            # -------------------------------------------------------------- #
            # sigma_schedule is stored as [B, T+1]; all rows are identical – take row 0.
            if "sigma_schedule" in data.batch.keys():
                sigma_schedule = data.batch["sigma_schedule"][0].float().to(device)  # [T+1]
            else:
                sampling_steps = actor_config.sampling_steps
                sigma_schedule = torch.linspace(0, 1, sampling_steps + 1, device=device)
                sigma_schedule = omni_time_shift(actor_config.shift, sigma_schedule)

            num_steps = sigma_schedule.shape[0] - 1   # T

            # Randomly select a fraction of timestep indices for training.
            timestep_fraction = actor_config.timestep_fraction
            n_train_timesteps = max(1, int(num_steps * timestep_fraction))
            train_timestep_indices = random.sample(range(num_steps), n_train_timesteps)

            # -------------------------------------------------------------- #
            # 4.  Rotary positional embeddings (frozen – same for all steps)  #
            # -------------------------------------------------------------- #
            freqs_cis = RotaryPosEmbedReal.get_freqs_real(
                self.dit.config.axes_dim_rope,
                self.dit.config.axes_lens,
                theta=10000,
            )

            # -------------------------------------------------------------- #
            # 5.  Compute GRPO advantages per group                           #
            # -------------------------------------------------------------- #
            batch_size = all_latents.shape[0]
            advantages = torch.zeros(batch_size, device=device, dtype=torch.float32)
            group_indices = torch.chunk(torch.arange(batch_size, device=device), batch_size // grpo_size)
            group_reward_stds = []
            for group_idx in group_indices:
                group_rewards = rewards[group_idx].float()
                grp_mean = group_rewards.mean()
                grp_std  = group_rewards.std().clamp_min(1e-8)
                advantages[group_idx] = (group_rewards - grp_mean) / grp_std
                group_reward_stds.append(group_rewards.std().item())  # raw std, before clamping
            avg_group_reward_std = sum(group_reward_stds) / len(group_reward_stds)
            min_group_reward_std = min(group_reward_stds)
            if torch.distributed.get_rank() == 0:
                logger.info(
                    f"[GRPO SIGNAL] group_reward_std(avg/min)={avg_group_reward_std:.5f}/{min_group_reward_std:.5f}  "
                    f"rewards={[f'{r:.4f}' for r in rewards.tolist()]}"
                )

            # -------------------------------------------------------------- #
            # 6.  Training loop                                                #
            # -------------------------------------------------------------- #
            self.dit.train()
            self.actor_optimizer.zero_grad()

            total_loss        = 0.0
            total_policy_loss = 0.0
            total_kl_loss     = 0.0

            # ---- Debug metric accumulators ---- #
            total_frac_clipped = 0.0      # fraction of ratios hitting clip boundary
            total_new_lp       = 0.0      # mean of recomputed new log-probs
            total_old_lp       = 0.0      # mean of old (rollout) log-probs
            total_lp_diff      = 0.0      # mean of (new - old) log-prob
            total_approx_kl    = 0.0      # approx KL = 0.5 * mean((ratio-1)^2)
            total_ratio_mean   = 0.0      # accumulated ratio mean (avg across all timesteps)
            total_ratio_max    = -float('inf')  # worst-case max ratio
            total_ratio_min    = float('inf')    # worst-case min ratio
            total_std_dev_t    = 0.0      # accumulate SDE noise scale per training timestep
            total_diff_abs     = 0.0      # accumulate |x_next - predicted_mean| per microbatch
            debug_count        = 0        # number of microbatches accumulated

            # Gradient-accumulation denominator: number of backward() calls.
            # mean() already averages over micro_bs samples, so the denominator
            # should count the number of accumulation steps, not total data points.
            n_microbatches_per_timestep = max(1, batch_size // micro_bs)
            grad_accum_denom = n_train_timesteps * n_microbatches_per_timestep

            train_step_count = 0
            for timestep_idx in train_timestep_indices:
                # Current and next noisy latents for this step.
                current_latents = all_latents[:, timestep_idx]      # [B, C, H, W]
                next_latents    = all_latents[:, timestep_idx + 1]  # [B, C, H, W]

                # Scalar timestep value (0–1) for DiT conditioning.
                t_val = sigma_schedule[timestep_idx]  # scalar
                # eta used during rollout, needed to recompute std_dev for SDE log-prob.
                t_next_val = sigma_schedule[timestep_idx + 1]
                dt   = (t_next_val - t_val).abs()
                eta  = getattr(self.config.rollout, "eta", 0.3)
                # Match rollout formula exactly: sqrt(abs(dt) + 1e-12) not sqrt(clamp(dt, 1e-12))
                std_dev_t = eta * torch.sqrt(torch.abs(torch.tensor(dt.item(), device=device, dtype=torch.float32)) + 1e-12)
                total_std_dev_t += std_dev_t.item()

                timestep_tensor = t_val.to(torch.bfloat16).expand(batch_size)  # [B] bf16, matching rollout

                start_time = time.time()
                # ---- microbatch loop ---- #
                mb_indices = torch.chunk(torch.arange(batch_size, device=device), max(1, batch_size // micro_bs))

                for mb_idx in mb_indices:
                    # Cast current latents to bf16 to match rollout precision:
                    # processing() casts prev_sample to bf16 between steps, so
                    # the rollout DiT and log-prob math use bf16-rounded values.
                    mb_current   = current_latents[mb_idx].to(torch.bfloat16)   # [mb, C, H, W] bf16
                    mb_next      = next_latents[mb_idx]                         # [mb, C, H, W] fp32
                    mb_text      = text_hidden_states[mb_idx].to(device)        # [mb, L, D]
                    mb_text_mask = text_attention_mask[mb_idx].to(device)       # [mb, L]
                    mb_adv       = advantages[mb_idx]                           # [mb]
                    mb_old_lp    = old_log_probs[mb_idx, timestep_idx]          # [mb]
                    mb_ts        = timestep_tensor[mb_idx]                      # [mb]

                    mb_neg_text      = neg_text_hidden_states[mb_idx].to(device)  if has_neg else None
                    mb_neg_text_mask = neg_text_attention_mask[mb_idx].to(device) if has_neg else None
                    mb_img_cond      = img_hidden_states[mb_idx].to(device)       if has_img_cond else None
                    mb_img_mask      = img_attention_mask[mb_idx].to(device)      if has_img_cond else None

                    # ---- Forward pass through DiT (with grad) ---- #
                    use_train_cfg = train_cfg_scale > 1.0 and has_neg and mb_neg_text is not None
                    with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
                        if use_train_cfg:
                            # Two separate passes to match rollout CFG exactly:
                            # Cond pass: positive text + image conditioning
                            cond_out = self.dit(
                                hidden_states=mb_current,
                                timestep=mb_ts,
                                text_hidden_states=mb_text,
                                text_attention_mask=mb_text_mask,
                                ar_image_hidden_states=mb_img_cond,
                                ar_image_attention_mask=mb_img_mask,
                                freqs_cis=freqs_cis,
                                return_dict=False,
                            )
                            if isinstance(cond_out, (list, tuple)):
                                cond_out = cond_out[0]
                            # Uncond pass: negative text, NO image conditioning
                            uncond_out = self.dit(
                                hidden_states=mb_current,
                                timestep=mb_ts,
                                text_hidden_states=mb_neg_text,
                                text_attention_mask=mb_neg_text_mask,
                                freqs_cis=freqs_cis,
                                return_dict=False,
                            )
                            if isinstance(uncond_out, (list, tuple)):
                                uncond_out = uncond_out[0]
                            model_output = uncond_out + train_cfg_scale * (cond_out - uncond_out)
                        else:
                            model_output = self.dit(
                                hidden_states=mb_current,
                                timestep=mb_ts,
                                text_hidden_states=mb_text,
                                text_attention_mask=mb_text_mask,
                                ar_image_hidden_states=mb_img_cond,
                                ar_image_attention_mask=mb_img_mask,
                                freqs_cis=freqs_cis,
                                return_dict=False,
                            )
                    # model_output is velocity prediction v; same shape as latents [mb, C, H, W]
                    if isinstance(model_output, (list, tuple)):
                        model_output = model_output[0]

                    # ---- Recompute log-prob of rollout trajectory under current policy ---- #
                    # prev_sample_mean  = x_t + dt * v  (Euler step mean)
                    prev_sample_mean = mb_current.to(torch.float32) + (t_next_val - t_val).item() * model_output.to(torch.float32)

                    # SDE correction term (DanceGRPO SDE solver, matching rollout)
                    x_hat = mb_current.to(torch.float32) + (1.0 - t_val.item()) * model_output.to(torch.float32)
                    score_estimate = -(mb_current.to(torch.float32) - t_val.item() * x_hat) / ((1.0 - t_val.item()) ** 2 + 1e-12)
                    prev_sample_mean = prev_sample_mean + 0.5 * (eta ** 2) * score_estimate * (t_next_val - t_val).item()

                    # Gaussian log-prob: log N(x_{t+1} | prev_sample_mean, std_dev_t^2 * I)
                    diff = mb_next.to(torch.float32) - prev_sample_mean
                    var_t    = (std_dev_t ** 2).clamp_min(1e-20)
                    log_std  = torch.log(std_dev_t.clamp_min(1e-20))
                    log_2pi  = torch.log(torch.tensor(2.0 * math.pi, device=device, dtype=torch.float32))
                    new_log_probs = (-(diff ** 2) / (2.0 * var_t) - log_std - 0.5 * log_2pi)
                    new_log_probs = new_log_probs.mean(dim=tuple(range(1, new_log_probs.ndim)))  # [mb]

                    # ---- PPO-clip loss ---- #
                    clamped_adv = torch.clamp(mb_adv, -adv_clip_max, adv_clip_max)
                    ratio       = torch.exp(new_log_probs - mb_old_lp.to(device))
                    ratio_mean  = ratio.mean().item()
                    ratio_max   = ratio.max().item()
                    ratio_min   = ratio.min().item()
                    unclipped   = -clamped_adv * ratio
                    clipped     = -clamped_adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    policy_loss = torch.mean(torch.maximum(unclipped, clipped)) / grad_accum_denom

                    loss = policy_loss

                    # ---- Debug metrics (no grad, cheap) ---- #
                    with torch.no_grad():
                        _is_clipped = (ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)
                        total_frac_clipped += _is_clipped.float().mean().item()
                        total_new_lp       += new_log_probs.mean().item()
                        total_old_lp       += mb_old_lp.mean().item()
                        total_lp_diff      += (new_log_probs - mb_old_lp.to(device)).mean().item()
                        total_approx_kl    += (0.5 * ((ratio - 1.0) ** 2)).mean().item()
                        total_ratio_mean   += ratio_mean
                        total_ratio_max    = max(total_ratio_max, ratio_max)
                        total_ratio_min    = min(total_ratio_min, ratio_min)
                        total_diff_abs     += diff.detach().abs().mean().item()
                        debug_count        += 1

                    # ---- Optional mean-prediction KL regulariser ---- #
                    # KL between N(mean_policy, sigma) and N(mean_ref, sigma) in closed form:
                    # KL = ||mean_policy - mean_ref||^2 / (2 sigma^2)
                    # (ref model would need a separate forward; omit if kl_coeff == 0)
                    if kl_coeff > 0.0 and self.ref_module_fsdp is not None:
                        with torch.no_grad():
                            with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
                                if use_train_cfg:
                                    cond_ref = self.ref_module_fsdp(
                                        hidden_states=mb_current,
                                        timestep=mb_ts,
                                        text_hidden_states=mb_text,
                                        text_attention_mask=mb_text_mask,
                                        ar_image_hidden_states=mb_img_cond,
                                        ar_image_attention_mask=mb_img_mask,
                                        freqs_cis=freqs_cis,
                                        return_dict=False,
                                    )
                                    if isinstance(cond_ref, (list, tuple)):
                                        cond_ref = cond_ref[0]
                                    uncond_ref = self.ref_module_fsdp(
                                        hidden_states=mb_current,
                                        timestep=mb_ts,
                                        text_hidden_states=mb_neg_text,
                                        text_attention_mask=mb_neg_text_mask,
                                        freqs_cis=freqs_cis,
                                        return_dict=False,
                                    )
                                    if isinstance(uncond_ref, (list, tuple)):
                                        uncond_ref = uncond_ref[0]
                                    ref_out = uncond_ref + train_cfg_scale * (cond_ref - uncond_ref)
                                else:
                                    ref_out = self.ref_module_fsdp(
                                        hidden_states=mb_current,
                                        timestep=mb_ts,
                                        text_hidden_states=mb_text,
                                        text_attention_mask=mb_text_mask,
                                        ar_image_hidden_states=mb_img_cond,
                                        ar_image_attention_mask=mb_img_mask,
                                        freqs_cis=freqs_cis,
                                        return_dict=False,
                                    )
                            if isinstance(ref_out, (list, tuple)):
                                ref_out = ref_out[0]
                            ref_mean = mb_current.to(torch.float32) + (t_next_val - t_val).item() * ref_out.to(torch.float32)

                        kl_loss = ((prev_sample_mean - ref_mean.detach()) ** 2).mean() / (2.0 * var_t) / grad_accum_denom
                        loss = policy_loss + kl_coeff * kl_loss

                        kl_reduced = kl_loss.detach().clone()
                        dist.all_reduce(kl_reduced, op=dist.ReduceOp.AVG)
                        total_kl_loss += kl_reduced.item()

                    # ---- Backward ---- #
                    loss.backward()

                    # ---- Track metrics ---- #
                    pl_reduced = policy_loss.detach().clone()
                    dist.all_reduce(pl_reduced, op=dist.ReduceOp.AVG)
                    total_policy_loss += pl_reduced.item()

                    l_reduced = loss.detach().clone()
                    dist.all_reduce(l_reduced, op=dist.ReduceOp.AVG)
                    total_loss += l_reduced.item()

                train_step_count += 1
                if torch.distributed.get_rank() == 0:
                    end_time = time.time()
                    logger.info(
                        f"update_actor step {train_step_count}/{n_train_timesteps} "
                        f"(t_idx={timestep_idx}, t={t_val.item()}, cfg={'Y' if use_train_cfg else 'N'}): "
                        f"loss={total_loss}  policy={total_policy_loss}  "
                        f"kl={total_kl_loss}  "
                        f"ratio(mean/max/min)={ratio_mean}/{ratio_max}/{ratio_min}  "
                        f"time={end_time - start_time}s"
                    )
                log_gpu_memory_usage(f"update_actor step {train_step_count}/{n_train_timesteps}", logger=logger, level=logging.DEBUG)

            # -------------------------------------------------------------- #
            # 7.  Gradient clip + optimiser step                              #
            # -------------------------------------------------------------- #
            grad_norm_raw = torch.nn.utils.clip_grad_norm_(self.dit.parameters(), max_grad_norm).item()
            grad_clip_ratio = grad_norm_raw / max(max_grad_norm, 1e-8)  # > 1.0 means clipping was active

            # Snapshot the parameter with the LARGEST gradient norm BEFORE optimizer step.
            _debug_param_name = None
            _debug_param_before = None
            if torch.distributed.get_rank() == 0:
                _best_gnorm = 0.0
                for pname, p in self.dit.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        gnorm = p.grad.float().norm().item()
                        if gnorm > _best_gnorm:
                            _best_gnorm = gnorm
                            _debug_param_name = pname
                            _debug_param_before = (p.data.float().norm().item(), gnorm, p.dtype)

            self.actor_optimizer.step()
            self.actor_lr_scheduler.step()

            # Check if parameters actually changed after the update.
            if torch.distributed.get_rank() == 0 and _debug_param_name is not None:
                old_norm, grad_norm_p, pdtype = _debug_param_before
                for pname2, p2 in self.dit.named_parameters():
                    if pname2 == _debug_param_name:
                        new_norm = p2.data.float().norm().item()
                        delta = abs(new_norm - old_norm)
                        logger.info(
                            f"[PARAM UPDATE CHECK] {_debug_param_name} dtype={pdtype}  "
                            f"norm_before={old_norm}  norm_after={new_norm}  "
                            f"delta={delta}  grad_norm={grad_norm_p}  "
                            f"grad_accum_denom={grad_accum_denom}"
                        )
                        break

            # ---- Post-step debug metrics ---- #

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # Average accumulated debug metrics
            _dc = max(debug_count, 1)
            avg_frac_clipped   = total_frac_clipped / _dc
            avg_new_lp         = total_new_lp / _dc
            avg_old_lp         = total_old_lp / _dc
            avg_lp_diff        = total_lp_diff / _dc
            avg_approx_kl      = total_approx_kl / _dc
            avg_ratio_mean     = total_ratio_mean / _dc
            avg_ratio_max      = total_ratio_max
            avg_ratio_min      = total_ratio_min
            avg_std_dev_t      = total_std_dev_t / max(n_train_timesteps, 1)
            avg_diff_abs       = total_diff_abs / max(debug_count, 1)
            advantage_mean     = advantages.mean().item()
            advantage_std      = advantages.std().item()
            reward_mean        = rewards.mean().item()
            reward_std         = rewards.std().item()
            reward_max         = rewards.max().item()
            reward_min         = rewards.min().item()

            metrics = {
                "train/total_loss":    total_loss,
                "train/policy_loss":   total_policy_loss,
                "train/kl_loss":       total_kl_loss,
                "train/lr":            self.actor_lr_scheduler.get_last_lr()[0],
                "train/ratio_mean":    avg_ratio_mean,   # avg across ALL timesteps (was last-step only)
                "train/ratio_max":     avg_ratio_max,    # worst-case max across all timesteps
                "train/ratio_min":     avg_ratio_min,    # worst-case min across all timesteps
                # ---- Debug diagnostics ---- #
                "debug/grad_norm_raw":      grad_norm_raw,
                "debug/grad_clip_ratio":    grad_clip_ratio,
                "debug/avg_group_reward_std": avg_group_reward_std,
                "debug/min_group_reward_std": min_group_reward_std,
                "debug/avg_std_dev_t":        avg_std_dev_t,
                "debug/avg_diff_abs":         avg_diff_abs,
                "debug/diff_over_noise":      avg_diff_abs / max(avg_std_dev_t, 1e-10),
                "debug/frac_clipped":       avg_frac_clipped,
                "debug/new_log_prob_mean":  avg_new_lp,
                "debug/old_log_prob_mean":  avg_old_lp,
                "debug/log_prob_diff_mean": avg_lp_diff,
                "debug/approx_kl":          avg_approx_kl,
                "debug/advantage_mean":     advantage_mean,
                "debug/advantage_std":      advantage_std,
                "debug/reward_mean":        reward_mean,
                "debug/reward_std":         reward_std,
                "debug/reward_max":         reward_max,
                "debug/reward_min":         reward_min,
            }

            if torch.distributed.get_rank() == 0:
                logger.info(
                    f"[DEBUG METRICS] "
                    f"grad_norm={grad_norm_raw}  "
                    f"ratio_avg(mean/max/min)={avg_ratio_mean}/{avg_ratio_max}/{avg_ratio_min}  "
                    f"frac_clipped={avg_frac_clipped}  "
                    f"log_prob(new/old/diff)={avg_new_lp}/{avg_old_lp}/{avg_lp_diff}  "
                    f"approx_kl={avg_approx_kl}  "
                    f"adv(mean/std)={advantage_mean}/{advantage_std}  "
                    f"reward(mean/std/max/min)={reward_mean}/{reward_std}/{reward_max}/{reward_min}  "
                    f"grpo_group_std(avg/min)={avg_group_reward_std:.5f}/{min_group_reward_std:.5f}  "
                    f"sde: std_dev_t={avg_std_dev_t:.5f}  diff_abs={avg_diff_abs:.5f}  "
                    f"diff/noise={avg_diff_abs / max(avg_std_dev_t, 1e-10):.1f}x  "
                    f"grad_clip_ratio={grad_clip_ratio:.2f}"
                )

            output = DataProto(meta_info={"metrics": metrics})
            log_gpu_memory_usage("After update_actor", logger=logger, level=logging.INFO)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
      
        # For mammothmoda25, we need to save each component separately
        if torch.distributed.get_rank() == 0:
            checkpoint_path = os.path.join(local_path, "actor_model.pt")
          
            # Save transformer and transformer_2 if they exist
            checkpoint = {}
            checkpoint['transformer'] = self.actor_module_fsdp.transformer.state_dict()
          
            if hasattr(self.actor_module_fsdp, 'transformer_2') and self.actor_module_fsdp.transformer_2 is not None:
                checkpoint['transformer_2'] = self.actor_module_fsdp.transformer_2.state_dict()
          
            # Save text encoder if it exists
            if hasattr(self.actor_module_fsdp, 'text_encoder') and self.actor_module_fsdp.text_encoder is not None:
                checkpoint['text_encoder'] = self.actor_module_fsdp.text_encoder.state_dict()
          
            torch.save(checkpoint, checkpoint_path)

        dist.barrier()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if torch.distributed.get_rank() == 0:
            logger.info(f"Saved checkpoint to local_path: {local_path}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )

        # No checkpoint to load, just offload the model and optimizer to CPU
        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
      
        # For mammothmoda25, load each component separately
        file_path = os.path.join(local_path, "actor_model.pt")
        checkpoint = torch.load(file_path)
      
        # Load transformer
        if 'transformer' in checkpoint:
            self.actor_module_fsdp.transformer.load_state_dict(checkpoint['transformer'])
      
        # Load transformer_2 if it exists
        if hasattr(self.actor_module_fsdp, 'transformer_2') and self.actor_module_fsdp.transformer_2 is not None:
            if 'transformer_2' in checkpoint:
                self.actor_module_fsdp.transformer_2.load_state_dict(checkpoint['transformer_2'])
      
        # Load text encoder if it exists
        if hasattr(self.actor_module_fsdp, 'text_encoder') and self.actor_module_fsdp.text_encoder is not None:
            if 'text_encoder' in checkpoint:
                self.actor_module_fsdp.text_encoder.load_state_dict(checkpoint['text_encoder'])

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)
        if torch.distributed.get_rank() == 0:
            logger.info(f"Loaded checkpoint from local_path: {local_path}")

    # ---------------------------------------------------------------------- #
    #  Fixed-evaluation helpers                                               #
    # ---------------------------------------------------------------------- #

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_fixed_eval(self, data: DataProto):
        """Initialise the fixed evaluation samples.

        Called once at the very start of training.  Picks up to 8 prompts from
        ``data``, runs the frozen MLLM for each to capture text conditioning,
        and creates a fixed initial noise tensor (seeded for reproducibility).
        Both the conditioning and the noise are stored per-prompt so that
        ``run_fixed_eval`` can render them after every weight update.

        All distributed ranks participate (required for FSDP2 MLLM forward).
        Rank 0's initial noise is broadcast to every other rank so all ranks
        share identical starting points.
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        try:
            raw_prompts = data.non_tensor_batch.get("prompts", None)
            if raw_prompts is None or len(raw_prompts) == 0:
                logger.warning("init_fixed_eval: no prompts found in data, skipping.")
                return DataProto()

            num_eval_prompts = min(8, len(raw_prompts))
            self._fixed_eval_samples = []  # list of dicts, one per prompt

            for p_idx in range(num_eval_prompts):
                prompt_str = str(raw_prompts[p_idx])
                single_prompt = np.array([prompt_str], dtype=object)
                single_data = DataProto.from_dict(
                    tensors={},
                    non_tensors={"prompts": single_prompt},
                )
                single_data.meta_info.update(data.meta_info)

                with torch.no_grad():
                    output = self.rollout._generate_minibatch(single_data)

                # ---- Initial noise -------------------------------------- #
                initial_noise = output.batch["all_latents"][0:1, 0].clone()  # [1, C, H, W]
                if torch.distributed.is_initialized():
                    dist.broadcast(initial_noise, src=0)

                # ---- Conditioning --------------------------------------- #
                conditioning = {}
                for key in [
                    "text_hidden_states",
                    "text_attention_mask",
                    "negative_text_hidden_states",
                    "negative_text_attention_mask",
                    "image_hidden_states",
                    "image_attention_mask",
                ]:
                    if key in output.batch.keys():
                        conditioning[key] = output.batch[key][0:1].clone().cpu()

                for len_key in ["text_seq_len", "neg_seq_len", "image_seq_len"]:
                    if len_key in output.batch.keys():
                        conditioning[len_key] = int(output.batch[len_key][0].item())

                # Sanitised short prompt tag for filenames (max 20 chars)
                safe_prompt = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt_str).strip("_")[:20] or "prompt"

                self._fixed_eval_samples.append({
                    "noise": initial_noise.cpu(),
                    "conditioning": conditioning,
                    "prompt": prompt_str,
                    "safe_prompt": safe_prompt,
                })

                if rank == 0:
                    logger.info(
                        f"Fixed eval [{p_idx+1}/{num_eval_prompts}] prompt: '{prompt_str[:80]}' | "
                        f"noise shape: {list(initial_noise.shape)}"
                    )

        except Exception as exc:
            logger.warning(f"init_fixed_eval failed: {exc}", exc_info=True)

        return DataProto()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def run_fixed_eval(self, data: DataProto):
        """Run deterministic inference with the fixed evaluation inputs.

        Called after every ``update_actor`` but only executes every 10 RL steps
        (controlled by ``global_step % 10 == 0``).  Uses the noise and MLLM
        conditioning captured in ``init_fixed_eval`` — both remain constant
        throughout training so visual progress is directly attributable to DiT
        weight changes.

        For each of the 8 fixed prompts, two images are generated using CFG
        scales 1.0 and 3.0, all starting from the identical fixed noise.
        The denoising loop is a pure Euler ODE (no SDE noise) for reproducible
        comparison.  Images are saved by rank 0 to:
          ``<output_dir>/<experiment_name>/fixed_eval/step_<N>_<prompt>_cfg<scale>.png``
        """
        if not hasattr(self, "_fixed_eval_samples") or not self._fixed_eval_samples:
            logger.debug("run_fixed_eval: not initialised, skipping.")
            return DataProto()

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        global_step = data.meta_info.get("global_step", 0)
        experiment_name = data.meta_info.get("experiment_name", "dance_grpo")

        # Only evaluate every 10 RL steps (step 0, 10, 20, …)
        if global_step % 10 != 0:
            return DataProto()

        output_dir = str(getattr(self.config.rollout, "output_dir", "logs"))
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)

        rollout_cfg = self.config.rollout
        num_inference_steps = int(getattr(rollout_cfg, "num_inference_steps", 40))
        h              = int(getattr(rollout_cfg, "height", 512))
        w              = int(getattr(rollout_cfg, "width", 512))
        vae_scale_factor = int(getattr(rollout_cfg, "vae_scale_factor", 16))

        eval_cfg_scales = [1.0, 3.0]

        try:
            from recipe.dance_grpo.mammothmoda2.model.mammothmoda2_dit import RotaryPosEmbedReal

            dev = get_device_name()

            # Rotary positional embeddings (frozen throughout training)
            freqs_cis = RotaryPosEmbedReal.get_freqs_real(
                self.dit.config.axes_dim_rope,
                self.dit.config.axes_lens,
                theta=10000,
            )

            # Warped timestep schedule — must match GRPOMockScheduler exactly.
            latent_h   = 2 * h // vae_scale_factor
            latent_w   = 2 * w // vae_scale_factor
            num_tokens = latent_h * latent_w
            timesteps  = torch.linspace(0, 1, num_inference_steps + 1)
            m          = (num_tokens ** 0.5) / 40.0
            timesteps  = timesteps / (m + timesteps * (1.0 - m))
            timesteps  = timesteps.to(dev)

            vae = self.full_model.gen_vae
            scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
            shift_factor   = getattr(vae.config, "shift_factor", 0.0)

            eval_dir = os.path.join(output_dir, experiment_name, "fixed_eval")
            if rank == 0:
                os.makedirs(eval_dir, exist_ok=True)

            self.dit.eval()

            for sample in self._fixed_eval_samples:
                cond = sample["conditioning"]
                safe_prompt = sample["safe_prompt"]

                # Strip left-padding and move conditioning to device
                text_hs = cond["text_hidden_states"]
                text_am = cond["text_attention_mask"]
                if "text_seq_len" in cond:
                    tl = cond["text_seq_len"]
                    text_hs = text_hs[:, -tl:, :]
                    text_am = text_am[:, -tl:]
                text_hs = text_hs.to(dev, dtype=torch.bfloat16)
                text_am = text_am.to(dev)

                has_neg = "negative_text_hidden_states" in cond
                neg_hs, neg_am = None, None
                if has_neg:
                    neg_hs = cond["negative_text_hidden_states"]
                    neg_am = cond["negative_text_attention_mask"]
                    if "neg_seq_len" in cond:
                        nl = cond["neg_seq_len"]
                        neg_hs = neg_hs[:, -nl:, :]
                        neg_am = neg_am[:, -nl:]
                    neg_hs = neg_hs.to(dev, dtype=torch.bfloat16)
                    neg_am = neg_am.to(dev)

                has_img = "image_hidden_states" in cond
                img_hs, img_am = None, None
                if has_img:
                    img_hs = cond["image_hidden_states"]
                    img_am = cond["image_attention_mask"]
                    if "image_seq_len" in cond:
                        il = cond["image_seq_len"]
                        img_hs = img_hs[:, -il:, :]
                        img_am = img_am[:, -il:]
                    img_hs = img_hs.to(dev, dtype=torch.bfloat16)
                    img_am = img_am.to(dev)

                for cfg_scale in eval_cfg_scales:
                    use_cfg = cfg_scale > 1.0 and has_neg

                    latents = sample["noise"].clone().to(dev, dtype=torch.bfloat16)

                    with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
                        for i in range(num_inference_steps):
                            t      = timesteps[i]
                            t_next = timesteps[i + 1]
                            dt     = (t_next - t).item()
                            ts     = t.unsqueeze(0).expand(1)

                            if use_cfg:
                                cond_out = self.dit(
                                    hidden_states=latents,
                                    timestep=ts,
                                    text_hidden_states=text_hs,
                                    text_attention_mask=text_am,
                                    ar_image_hidden_states=img_hs,
                                    ar_image_attention_mask=img_am,
                                    freqs_cis=freqs_cis,
                                    return_dict=False,
                                )
                                if isinstance(cond_out, (list, tuple)):
                                    cond_out = cond_out[0]
                                uncond_out = self.dit(
                                    hidden_states=latents,
                                    timestep=ts,
                                    text_hidden_states=neg_hs,
                                    text_attention_mask=neg_am,
                                    freqs_cis=freqs_cis,
                                    return_dict=False,
                                )
                                if isinstance(uncond_out, (list, tuple)):
                                    uncond_out = uncond_out[0]
                                model_output = uncond_out + cfg_scale * (cond_out - uncond_out)
                            else:
                                model_output = self.dit(
                                    hidden_states=latents,
                                    timestep=ts,
                                    text_hidden_states=text_hs,
                                    text_attention_mask=text_am,
                                    ar_image_hidden_states=img_hs,
                                    ar_image_attention_mask=img_am,
                                    freqs_cis=freqs_cis,
                                    return_dict=False,
                                )
                                if isinstance(model_output, (list, tuple)):
                                    model_output = model_output[0]

                            latents = latents + dt * model_output.to(torch.bfloat16)

                    # ---- Decode with VAE -------------------------------- #
                    latents_to_decode = latents.float() / scaling_factor + shift_factor

                    with torch.no_grad():
                        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
                            decoded = vae.decode(latents_to_decode.to(torch.bfloat16)).sample

                    decoded = decoded.float().cpu()
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    decoded = decoded.permute(0, 2, 3, 1).numpy()
                    decoded = (decoded * 255).round().astype(np.uint8)
                    pil_image = Image.fromarray(decoded[0])

                    if rank == 0:
                        cfg_tag = f"cfg{cfg_scale:.1f}"
                        img_path = os.path.join(
                            eval_dir,
                            f"step_{global_step:06d}_{safe_prompt}_{cfg_tag}.png",
                        )
                        pil_image.save(img_path)
                        logger.info(f"Fixed eval image saved → {img_path}")

        except Exception as exc:
            logger.warning(f"run_fixed_eval step {global_step} failed: {exc}", exc_info=True)
        finally:
            self.dit.train()

        return DataProto()


# Helper functions moved from train_grpo_edit.py
def omni_time_shift(shift, t):
    t = 1 - t
    t = (shift * t) / (1 + (shift - 1) * t)
    t = 1 - t
    return t

