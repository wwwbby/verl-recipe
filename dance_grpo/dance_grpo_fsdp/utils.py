import json
import os

import numpy as np
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardedStateDictConfig
from torch.distributed.fsdp.api import (ShardedStateDictConfig,
                                        ShardingStrategy, StateDictType)

from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (CPUOffloadPolicy, MixedPrecisionPolicy,
                                   apply_fsdp2, collect_lora_params,
                                   fsdp2_load_full_state_dict, fsdp_version,
                                   get_fsdp_wrap_policy,
                                   get_init_weight_context_manager,
                                   get_shard_placement_fn, init_fn,
                                   layered_summon_lora_params,
                                   load_fsdp_model_to_gpu, load_fsdp_optimizer,
                                   offload_fsdp_model_to_cpu,
                                   offload_fsdp_optimizer,
                                   replace_lora_wrapper)
from verl.utils.torch_dtypes import PrecisionType


def count_parameters(model):
    """计算模型总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  
    return total_params, trainable_params


def format_params(num):
    """将参数量转换为更易读的格式"""
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}T"


def calculate_and_print_model_params(model, verbose=True):
    """计算并打印模型参数量和模型大小
  
    Args:
        model (torch.nn.Module): 要计算参数量的模型
        verbose (bool): 是否打印详细信息
      
    Returns:
        tuple: (总参数量, 可训练参数量)
    """
    total_params, trainable_params = count_parameters(model)
  
    if verbose:
        print(f"总参数量: {total_params:,} ({format_params(total_params)})")
        print(f"可训练参数量: {trainable_params:,} ({format_params(trainable_params)})")
        print(f"模型大小 (FP32): {total_params * 4 / (1024 ** 3):.2f} GB")
        print(f"模型大小 (BF16): {total_params * 2 / (1024 ** 3):.2f} GB")
  
    return total_params, trainable_params


def load_repeat_data(file_index, grpo_size, root_dir='./data'):
    # load single data
    file_index = str(file_index)
    ref_latent = torch.load(os.path.join(root_dir, 'visual_embed', file_index))
    prompt_embed = torch.load(os.path.join(root_dir, 'prompt_embed', file_index)).to('npu')
    prompt_attention_mask = torch.load(os.path.join(root_dir, 'prompt_attention_mask', file_index)).to('npu')
    negative_prompt_embed = torch.load(os.path.join(root_dir, 'negative_prompt_embed', file_index)).to('npu')
    negative_prompt_attention_mask = torch.load(os.path.join(root_dir, 'negative_prompt_attention_mask', file_index)).to('npu')
  
    unqueeze_tensor = lambda i : i.unsqueeze(0) if i.shape[0] != 1 else i
    prompt_embed, prompt_attention_mask, negative_prompt_embed, negative_prompt_attention_mask = \
        unqueeze_tensor(prompt_embed), unqueeze_tensor(prompt_attention_mask), unqueeze_tensor(negative_prompt_embed), unqueeze_tensor(negative_prompt_attention_mask)
    prompt_embed = torch.repeat_interleave(prompt_embed, grpo_size, dim=0)
    prompt_attention_mask = torch.repeat_interleave(prompt_attention_mask, grpo_size, dim=0)
    negative_prompt_embed = torch.repeat_interleave(negative_prompt_embed, grpo_size, dim=0)
    negative_prompt_attention_mask = torch.repeat_interleave(negative_prompt_attention_mask, grpo_size, dim=0)
    ref_latent = [unqueeze_tensor(i.to('npu')) for i in ref_latent]*grpo_size
    return ref_latent, prompt_embed, prompt_attention_mask, negative_prompt_embed, negative_prompt_attention_mask


def init_fsdp_module(module, strategy="fsdp2", process_group=None, only_forward=False):
    param_dtype = PrecisionType.to_dtype("bf16")
    reduce_dtype = PrecisionType.to_dtype("fp32")
    buffer_dtype = PrecisionType.to_dtype("fp32")

    mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

    auto_wrap_policy = get_fsdp_wrap_policy(
        module=module,
        config=None,
        is_lora=False
    )
    import torch.distributed as dist
    world_size = dist.get_world_size()
    fsdp_mesh = init_device_mesh("npu", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    from torch.distributed.fsdp import ShardingStrategy
    sharding_strategy = ShardingStrategy.FULL_SHARD

    # Note: We force turn off CPUOffload because it causes incorrect results when using grad accumulation
    if strategy == "fsdp":
        # cpu_offload:
        # - actor: None
        # - critic: None
        # - ref: CPUOffload(offload_params=True)
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        offload_policy = None
        if only_forward:
            offload_policy = CPUOffload(offload_params=True)
        module = FSDP(
            module,
            param_init_fn=init_fn,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            process_group=process_group,
            device_mesh=fsdp_mesh,
            forward_prefetch=True,
            use_orig_params=True,
            cpu_offload=offload_policy,
        )
    elif strategy == "fsdp2":
        # - actor: offload_policy
        # - critic: offload_policy
        # - ref: CPUOffloadPolicy(pin_memory=True)
        assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
        )

        offload_policy = None
        if only_forward:
            offload_policy = CPUOffloadPolicy(pin_memory=True)

        fsdp_kwargs = {
            "mesh": fsdp_mesh,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
            "reshard_after_forward": False,
        }
        full_state = module.state_dict()
        module = module.npu()
        apply_fsdp2(module, fsdp_kwargs, {})
        fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)
    else:
        raise NotImplementedError(f"Unknown strategy {strategy}")

    # if self.model_config.enable_activation_offload:
    #     enable_gradient_checkpointing = self.model_config.enable_gradient_checkpointing
    #     enable_activation_offloading(module, self.engine_config.strategy, enable_gradient_checkpointing)

    if torch.distributed.get_world_size() == 1 and fsdp_version(module) == 1:
        FSDP.set_state_dict_type(
            module,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(),
        )
    elif fsdp_version(module) == 1:
        FSDP.set_state_dict_type(
            module,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(),
        )
  
    # module.enable_gradient_checkpointing()
    return module
