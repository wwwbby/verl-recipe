# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import os
import math

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, actor_module: nn.Module, config: ActorConfig):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.config = config
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(config.get("dtype", "bfloat16"))

    def forward_micro_batch(
        self,
        latents,
        pre_latents,
        ref_latents,
        text_hidden_states,
        text_attention_mask,
        i,
        negative_text_hidden_states=None,
        negative_text_attention_mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        GRPO one step implementation for diffusion model training
        
        Args:
            data: DataProto containing batch data
            latents: Current latents
            pre_latents: Previous latents
            text_hidden_states: Text embeddings for conditioning
            text_attention_mask: Attention mask for text embeddings
            timestep: Current timestep
            timesteps_full: Full timestep schedule
            i: Current step index
            ref_latents: Reference latents for guidance
            negative_text_hidden_states: Negative prompts for CFG
            negative_text_attention_mask: Attention mask for negative prompts
            
        Returns:
            log_prob: Log probability for PPO bookkeeping
            prev_sample_mean: Mean of the previous sample
            std_dev_t: Standard deviation at timestep t
        """
        # Forward pass through the actor module
        is_test = os.getenv("IS_TEST", "TRUE").upper() in ["TRUE", "1"]
        if is_test:
            text_hidden_states = torch.randn(
            (text_hidden_states.shape[0], text_hidden_states.shape[1], 3584),
            device=text_hidden_states.device,
            dtype=torch.bfloat16,
            )
            negative_text_hidden_states = torch.randn(
                (negative_text_hidden_states.shape[0], negative_text_hidden_states.shape[1], 3584),
                device=negative_text_hidden_states.device,
                dtype=torch.bfloat16,
            )
        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            log_probs, prev_sample_mean, std_dev_t = self.actor_module.dit_foward(
                latents,
                pre_latents,
                text_hidden_states,
                text_attention_mask,
                ref_latents,
                negative_text_hidden_states,
                negative_text_attention_mask,
                i
            )
        return log_probs, prev_sample_mean, std_dev_t

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        pass

    def update_policy(self, data: DataProto) -> dict:
        pass