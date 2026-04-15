import argparse
import math
import multiprocessing
import os
import random
import time
from collections import deque

import torch
import torch.distributed as dist
import wandb
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_mammoth2_rl_imgedit_datasets import LatentEditDataset, latent_collate_function
from fastvideo.rewards.editscore.editscore_reward_model import create_editscore_reward_model
from fastvideo.utils.checkpoint import (
    save_checkpoint,
)
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper_imgedit
from fastvideo.utils.ema import FSDP_EMA, save_ema_checkpoint
from fastvideo.utils.fsdp_util import apply_fsdp_checkpointing, get_dit_fsdp_kwargs
from fastvideo.utils.logging_ import main_print
from fastvideo.utils.parallel_states import (
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
)
from mammoth2.models.attention_processor import USE_REAL_ROPE
from mammoth2.models.transformers.rope import RotaryPosEmbed
from mammoth2.models.transformers.rope_real import RotaryPosEmbedReal

# Mammoth2 imports
from mammoth2.models.transformers.transformer_dit import Transformer2DModel
from mammoth2.models.unified_mammoth2model import replace_rmsnorm_with_custom
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

check_min_version("0.31.0")


def move_to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)


def omni_time_shift(shift, t):
    t = 1 - t
    t = (shift * t) / (1 + (shift - 1) * t)
    t = 1 - t
    return t


def compute_eta_t(
    eta: float,
    t: torch.Tensor,
    schedule: str = "constant",
    power: float = 2.0,
    min_ratio: float = 0.0,
):
    if schedule == "constant":
        scale = torch.ones_like(t)
    elif schedule == "linear":
        scale = 1.0 - t
    elif schedule == "cosine":
        scale = 0.5 * (1.0 + torch.cos(torch.pi * t))
    elif schedule == "poly":
        scale = torch.clamp(1.0 - t, min=0.0) ** power
    else:
        scale = torch.ones_like(t)
    if min_ratio > 0.0:
        scale = torch.clamp(scale, min=min_ratio)
    return torch.as_tensor(eta, dtype=t.dtype, device=t.device) * scale


def mammoth2_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    timesteps: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
    eps_schedule: str,
    eps_power: float,
    eps_min_ratio: float,
    generators: list | None = None,
):
    t = timesteps[index]
    t_next = timesteps[index + 1]
    dt = t_next - t
    prev_sample_mean = latents + dt * model_output
    x_hat = latents + (1.0 - t) * model_output
    pred_original_sample = x_hat
    score_estimate = -(latents - t * x_hat) / ((1.0 - t) ** 2 + 1e-12)

    eta_t = compute_eta_t(eta, t, schedule=eps_schedule, power=eps_power, min_ratio=eps_min_ratio)
    std_dev_t = eta_t * torch.sqrt(torch.abs(dt) + 1e-12)

    if sde_solver:
        prev_sample_mean = prev_sample_mean + (-0.5 * (eta_t**2)) * score_estimate * dt

    if grpo and prev_sample is None:
        if generators is not None and len(generators) == prev_sample_mean.shape[0]:
            from diffusers.utils.torch_utils import randn_tensor

            noise = randn_tensor(
                prev_sample_mean.shape,
                generator=generators,
                device=prev_sample_mean.device,
                dtype=prev_sample_mean.dtype,
            )
        else:
            noise = torch.randn_like(prev_sample_mean)
        prev_sample = prev_sample_mean + noise * std_dev_t

    if grpo:
        diff = prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)
        std32 = std_dev_t.to(torch.float32)
        var_t = (std32**2).clamp_min(1e-20)
        log_std = torch.log(std32.clamp_min(1e-20))
        log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=std_dev_t.device, dtype=torch.float32))
        log_prob = (-(diff**2) / (2 * var_t)) - log_std - 0.5 * log_two_pi
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample


def mammoth2_step_aligned(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    timesteps: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
    eps_schedule: str,
    eps_power: float,
    eps_min_ratio: float,
    generators: list | None = None,
):
    t = timesteps[index]
    t_next = timesteps[index + 1]
    dt = t_next - t
    prev_sample_mean = latents + dt * model_output
    x_hat = latents + (1.0 - t) * model_output
    pred_original_sample = x_hat
    score_estimate = -(latents - t * x_hat) / ((1.0 - t) ** 2 + 1e-12)

    # compute_eta_t(eta, t, schedule=eps_schedule, power=eps_power, min_ratio=eps_min_ratio)
    eta_t = eta
    std_dev_t = eta_t * torch.sqrt(torch.abs(dt) + 1e-12)

    if sde_solver:
        prev_sample_mean = prev_sample_mean + (0.5 * (eta_t**2)) * score_estimate * dt

    if grpo and prev_sample is None:
        if generators is not None and len(generators) == prev_sample_mean.shape[0]:
            from diffusers.utils.torch_utils import randn_tensor

            noise = randn_tensor(
                prev_sample_mean.shape,
                generator=generators,
                device=prev_sample_mean.device,
                dtype=prev_sample_mean.dtype,
            )
        else:
            noise = torch.randn_like(prev_sample_mean)
        prev_sample = prev_sample_mean + noise * std_dev_t

    if grpo:
        diff = prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)
        std32 = std_dev_t.to(torch.float32)
        var_t = (std32**2).clamp_min(1e-20)
        log_std = torch.log(std32.clamp_min(1e-20))
        log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=std_dev_t.device, dtype=torch.float32))
        log_prob = (-(diff**2) / (2 * var_t)) - log_std - 0.5 * log_two_pi
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob, prev_sample_mean, std_dev_t
    else:
        return prev_sample_mean, pred_original_sample


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def build_freqs_cis(transformer: Transformer2DModel, device: torch.device):
    if USE_REAL_ROPE:
        freqs_cis = RotaryPosEmbedReal.get_freqs_real(
            transformer.config.axes_dim_rope,
            transformer.config.axes_lens,
            theta=10000,
        )
    else:
        freqs_cis = RotaryPosEmbed.get_freqs_cis(
            transformer.config.axes_dim_rope,
            transformer.config.axes_lens,
            theta=10000,
        )
    return freqs_cis


def run_sample_step(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    text_hidden_states,
    text_attention_mask,
    ref_latent,
    freqs_cis,
    grpo_sample,
    generators=None,
    negative_text_hidden_states=None,
    negative_text_attention_mask=None,
    sde_sample: bool = False,
):
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:
            B = text_hidden_states.shape[0]
            t = sigma_schedule[i]
            t_next = sigma_schedule[i + 1]
            dt = t_next - t
            timestep_value = int(t * 1000)
            timestep_in = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
            transformer.eval()
            with torch.autocast("npu", torch.bfloat16):
                # Per-batch isolation (align with sample_mammoth2.py)
                z = z.contiguous().clone()
                text_hidden_states = text_hidden_states.contiguous().clone()
                text_attention_mask = text_attention_mask.to(torch.bool).contiguous().clone()
                optional_kwargs = {}
                optional_kwargs["ref_image_hidden_states"] = ref_latent

                if bool(getattr(args, "use_negative_cfg", 1)):
                    pred_c = transformer(
                        z,
                        (timestep_in.to(z.dtype) / 1000).contiguous().clone(),
                        text_hidden_states,
                        text_attention_mask,
                        freqs_cis=freqs_cis,
                        **optional_kwargs,
                    )
                    use_neg = (negative_text_hidden_states is not None) and (negative_text_attention_mask is not None)
                    uncond_states = (
                        (negative_text_hidden_states if use_neg else torch.zeros_like(text_hidden_states))
                        .contiguous()
                        .clone()
                    )
                    uncond_mask = (
                        (
                            (
                                negative_text_attention_mask
                                if use_neg
                                else torch.zeros_like(text_attention_mask, dtype=torch.bool)
                            ).to(torch.bool)
                        )
                        .contiguous()
                        .clone()
                    )
                    pred_u = transformer(
                        z,
                        (timestep_in.to(z.dtype) / 1000).contiguous().clone(),
                        uncond_states,
                        uncond_mask,
                        freqs_cis=freqs_cis,
                    )

                    s = float(getattr(args, "cfg_scale", 5.0))
                    s_image = float(getattr(args, "cfg_scale_image", 2.0))

                    if s_image > 0.0:
                        pred_ref = transformer(
                            z,
                            (timestep_in.to(z.dtype) / 1000).contiguous().clone(),
                            uncond_states,
                            uncond_mask,
                            freqs_cis=freqs_cis,
                            **optional_kwargs,
                        )

                        pred = pred_u + s * (pred_c - pred_ref) + s_image * (pred_ref - pred_u)
                    else:
                        pred = pred_u + s * (pred_c - pred_u)
                else:
                    pred = transformer(
                        z,
                        (timestep_in.to(z.dtype) / 1000).contiguous().clone(),
                        text_hidden_states,
                        text_attention_mask,
                        freqs_cis=freqs_cis,
                        **optional_kwargs,
                    )
            if sde_sample:
                # Use SDE step aligned with mammoth2 SDE
                z, pred_original, log_prob, prev_sample_mean, std_dev_t = mammoth2_step_aligned(
                    pred,
                    z.to(torch.float32),
                    args.eta,
                    sigma_schedule,
                    i,
                    prev_sample=None,
                    grpo=True,
                    sde_solver=True,
                    eps_schedule=args.eps_schedule,
                    eps_power=args.eps_power,
                    eps_min_ratio=args.eps_min_ratio,
                    generators=generators,
                )
                z = z.to(torch.bfloat16)
                all_latents.append(z)
                all_log_probs.append(log_prob)
            else:
                # Deterministic Euler step to align with SRPO sampling
                z = (z.to(torch.float32) + dt * pred.to(torch.float32)).to(torch.bfloat16)

                # Compute a consistent per-step log_prob under zero-noise assumption for PPO bookkeeping
                eta_t = compute_eta_t(
                    args.eta,
                    torch.as_tensor(t, device=z.device, dtype=torch.float32),
                    schedule=args.eps_schedule,
                    power=args.eps_power,
                    min_ratio=args.eps_min_ratio,
                )
                std_dev_t = eta_t * torch.sqrt(
                    torch.abs(torch.as_tensor(dt, device=z.device, dtype=torch.float32)) + 1e-12
                )
                prev_mean = all_latents[-1].to(torch.float32) + dt * pred.to(torch.float32)
                prev_sample = prev_mean  # zero noise path
                diff = prev_sample - prev_mean
                std32 = std_dev_t.to(torch.float32)
                var_t = (std32**2).clamp_min(1e-20)
                log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=std_dev_t.device, dtype=torch.float32))
                # elementwise log_prob then sum over non-batch dims
                elem_log_prob = (-(diff**2) / (2 * var_t)) - torch.log(std32.clamp_min(1e-20)) - 0.5 * log_two_pi
                log_prob = elem_log_prob.mean(dim=tuple(range(1, elem_log_prob.ndim)))

                all_latents.append(z)
                all_log_probs.append(log_prob)
        latents = pred_original if sde_sample else z
        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1)
        return z, latents, all_latents, all_log_probs


def grpo_one_step(
    args,
    latents,
    pre_latents,
    text_hidden_states,
    text_attention_mask,
    transformer,
    timestep,
    timesteps_full,
    i,
    ref_latents,
    negative_text_hidden_states=None,
    negative_text_attention_mask=None,
):
    B = text_hidden_states.shape[0]
    transformer.train()
    with torch.autocast("npu", torch.bfloat16):
        freqs_cis = build_freqs_cis(transformer, latents.device)
        timestep_in = timestep.repeat(B).to(latents.dtype).clone() / 1000
        optional_kwargs = {}
        optional_kwargs["ref_image_hidden_states"] = ref_latents

        if bool(getattr(args, "use_negative_cfg", 1)):
            pred_c = transformer(
                latents,
                timestep_in,
                text_hidden_states,
                text_attention_mask.to(torch.bool),
                freqs_cis=freqs_cis,
                **optional_kwargs,
            )
            use_neg = (negative_text_hidden_states is not None) and (negative_text_attention_mask is not None)
            uncond_states = negative_text_hidden_states if use_neg else torch.zeros_like(text_hidden_states)
            uncond_mask = (
                negative_text_attention_mask if use_neg else torch.zeros_like(text_attention_mask, dtype=torch.bool)
            ).to(torch.bool)
            pred_u = transformer(
                latents,
                timestep_in,
                uncond_states,
                uncond_mask,
                freqs_cis=freqs_cis,
            )

            s = float(getattr(args, "cfg_scale", 5.0))
            s_image = float(getattr(args, "cfg_scale_image", 2.0))

            if s_image > 0.0:
                pred_ref = transformer(
                    latents,
                    timestep_in,
                    uncond_states,
                    uncond_mask,
                    freqs_cis=freqs_cis,
                    **optional_kwargs,
                )

                pred = pred_u + s * (pred_c - pred_ref) + s_image * (pred_ref - pred_u)
            else:
                pred = pred_u + s * (pred_c - pred_u)
        else:
            pred = transformer(
                latents,
                timestep_in,
                text_hidden_states,
                text_attention_mask.to(torch.bool),
                freqs_cis=freqs_cis,
                **optional_kwargs,
            )
    # Choose deterministic vs SDE log_prob depending on eta
    sde_sampling = (args.eta is not None) and (float(args.eta) > 0.0)
    if sde_sampling:
        _, _, log_prob, prev_sample_mean, std_dev_t = mammoth2_step_aligned(
            pred,
            latents.to(torch.float32),
            args.eta,
            timesteps_full,
            i,
            prev_sample=pre_latents.to(torch.float32),
            grpo=True,
            sde_solver=True,
            eps_schedule=args.eps_schedule,
            eps_power=args.eps_power,
            eps_min_ratio=args.eps_min_ratio,
            generators=None,
        )
        return log_prob, prev_sample_mean, std_dev_t
    else:
        # Deterministic ODE path: compute zero-noise log_prob like sampling
        t = timesteps_full[i]
        t_next = timesteps_full[i + 1]
        dt = t_next - t
        prev_mean = latents.to(torch.float32) + dt * pred.to(torch.float32)
        prev_sample = pre_latents.to(torch.float32)
        eta_t = compute_eta_t(
            args.eta if args.eta is not None else 0.0,
            torch.as_tensor(t, device=latents.device, dtype=torch.float32),
            schedule=args.eps_schedule,
            power=args.eps_power,
            min_ratio=args.eps_min_ratio,
        )
        std_dev_t = eta_t * torch.sqrt(
            torch.abs(torch.as_tensor(dt, device=latents.device, dtype=torch.float32)) + 1e-12
        )
        diff = prev_sample - prev_mean
        std32 = std_dev_t.to(torch.float32)
        var_t = (std32**2).clamp_min(1e-20)
        log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=std_dev_t.device, dtype=torch.float32))
        elem_log_prob = (-(diff**2) / (2 * var_t)) - torch.log(std32.clamp_min(1e-20)) - 0.5 * log_two_pi
        log_prob = elem_log_prob.mean(dim=tuple(range(1, elem_log_prob.ndim)))
        return log_prob


def calculate_rewards_async(args_tuple):
    global reward_model
    start_time = time.time()
    rank = int(os.environ["RANK"])
    prompts, images = args_tuple
    # Ask hybrid reward model for detailed metadata so we can log per-component rewards
    rewards_with_meta = reward_model.batch_calculate_rewards(prompts, images, return_metadata=True)

    # rewards_with_meta -> (rewards: List[float], metadata_list: List[Dict])
    if isinstance(rewards_with_meta, tuple) and len(rewards_with_meta) == 2:
        hybrid_rewards, metadata_list = rewards_with_meta
    else:
        hybrid_rewards, metadata_list = rewards_with_meta, None

    rewards_cpu = [float(r) for r in hybrid_rewards]
    metadata_cpu = metadata_list if metadata_list is not None else [{} for _ in rewards_cpu]
    end_time = time.time()
    if rank == 0:
        print(f"[Reward] Batch time: {end_time - start_time}")

    return rewards_cpu, metadata_cpu


reward_model = None


def init_reward_model_worker(psm, num_pass):
    global reward_model
    reward_model = create_editscore_reward_model(psm=psm, num_pass=num_pass)


def sample_reference_model(
    args,
    device,
    transformer,
    vae,
    text_hidden_states,
    text_attention_masks,
    ref_latents,
    tokenizer,
    edit_prompts,
    preprocess_val,
    images_dir,
    sample_batch_size,
    ref_images,
    # optional negatives for CFG
    negative_text_hidden_states=None,
    negative_text_attention_masks=None,
):
    sample_steps = args.sampling_steps
    # Use forward-time schedule t in [0,1] to match SRPO sampling flow
    sigma_schedule = torch.linspace(0, 1, sample_steps + 1).to(device)
    sigma_schedule = omni_time_shift(args.shift, sigma_schedule)
    freqs_cis = build_freqs_cis(transformer, device)
    pool = multiprocessing.Pool(
        processes=args.reward_num_workers,
        initializer=init_reward_model_worker,
        initargs=(args.editscore_psm, args.editscore_num_pass),
    )

    image_processor = VaeImageProcessor(16)
    rank = int(os.environ["RANK"])

    all_latents = []
    all_log_probs = []
    all_rewards = []
    prompt_following_rewards = []
    consistency_rewards = []
    perceptual_quality_rewards = []

    for sample_idx, (
        text_hidden_state,
        text_attention_mask,
        ref_latent,
        edit_prompt,
        ref_image,
    ) in enumerate(zip(text_hidden_states, text_attention_masks, ref_latents, edit_prompts, ref_images, strict=True)):
        B = text_hidden_state.shape[0]
        IN_CHANNELS, latent_w, latent_h = ref_latent[0].shape[1], ref_latent[0].shape[2], ref_latent[0].shape[3]

        num_chunks = max(1, (B + sample_batch_size - 1) // sample_batch_size)
        batch_indices = torch.chunk(torch.arange(B), num_chunks)
        edit_prompt_list = [edit_prompt] * B
        if args.init_same_noise:
            input_latents = (
                torch.randn(
                    (1, IN_CHANNELS, latent_w, latent_h),
                    device=device,
                    dtype=torch.bfloat16,
                )
                .contiguous()
                .clone()
            )
            input_latents = input_latents.repeat(sample_batch_size, 1, 1, 1).contiguous().clone()

        _all_latents = []
        _all_log_probs = []
        _all_rewards = []
        reward_futures = []

        for index, batch_idx in enumerate(batch_indices):
            start_time = time.time()
            batch_text_hidden_state = text_hidden_state[batch_idx].contiguous().clone()
            batch_text_attention_mask = text_attention_mask[batch_idx].to(torch.bool).contiguous().clone()
            if negative_text_hidden_states is not None and negative_text_hidden_states[0] is not None:
                batch_negative_text_hidden_state = (
                    negative_text_hidden_states[sample_idx][batch_idx].contiguous().clone()
                )
                batch_negative_text_attention_mask = (
                    negative_text_attention_masks[sample_idx][batch_idx].to(torch.bool).contiguous().clone()
                )
            else:
                batch_negative_text_hidden_state = None
                batch_negative_text_attention_mask = None

            batch_indices_list = batch_idx.tolist()
            batch_ref_latent = [ref_latent[i] for i in batch_indices_list]
            batch_edit_prompt = [edit_prompt_list[int(i)] for i in batch_indices_list]
            if not args.init_same_noise:
                input_latents = (
                    torch.randn(
                        (len(batch_idx), IN_CHANNELS, latent_w, latent_h),
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    .contiguous()
                    .clone()
                )

            grpo_sample = True
            if args.show_progress_bar:
                progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
            else:
                progress_bar = range(0, sample_steps)

            with torch.no_grad():
                z, latents, batch_latents, batch_log_probs = run_sample_step(
                    args,
                    input_latents,
                    progress_bar,
                    sigma_schedule,
                    transformer,
                    batch_text_hidden_state,
                    batch_text_attention_mask,
                    batch_ref_latent,
                    freqs_cis,
                    grpo_sample,
                    generators=None,
                    negative_text_hidden_states=(
                        batch_negative_text_hidden_state if batch_negative_text_hidden_state is not None else None
                    ),
                    negative_text_attention_mask=(
                        batch_negative_text_attention_mask if batch_negative_text_attention_mask is not None else None
                    ),
                    sde_sample=True,
                )

            _all_latents.append(batch_latents)
            _all_log_probs.append(batch_log_probs)

            with torch.inference_mode():
                with torch.autocast("npu", dtype=torch.bfloat16):
                    # Use VAE config scaling/shift if available; otherwise, do not alter latents
                    latents_to_decode = latents
                    scaling_factor = getattr(vae.config, "scaling_factor", None)
                    shift_factor = getattr(vae.config, "shift_factor", None)
                    if scaling_factor is not None:
                        latents_to_decode = latents_to_decode / scaling_factor
                    if shift_factor is not None:
                        latents_to_decode = latents_to_decode + shift_factor
                    image = vae.decode(latents_to_decode, return_dict=False)[0]
                    decoded_image = image_processor.postprocess(image)

            for i in range(len(decoded_image)):
                save_path = f"{images_dir}/m2_{rank}_{index}_{i}.png"
                decoded_image[i].save(save_path)
                if rank == 0:
                    print(f"[Sampling] Saved image: {save_path}")
            end_time = time.time()
            if rank == 0:
                print(f"[Sampling] Batch {index} time: {end_time - start_time}")

            prompts = batch_edit_prompt
            images = [[ref_image, _decoded_image] for _decoded_image in decoded_image]

            args_tuple = (prompts, images)
            reward_futures.append(pool.apply_async(calculate_rewards_async, (args_tuple,)))

        for future in reward_futures:
            rewards_cpu, metadata_list = future.get()
            for idx_r, reward_score in enumerate(rewards_cpu):
                score = torch.tensor(reward_score, device=device, dtype=torch.float32).unsqueeze(0)
                _all_rewards.append(score)
                meta = metadata_list[idx_r] if idx_r < len(metadata_list) else {}
                prompt_following = float(meta.get("prompt_following", 0.0))
                consistency = float(meta.get("consistency", 0.0))
                perceptual = float(meta.get("perceptual_quality", 0.0))
                prompt_following_rewards.append(
                    torch.tensor(prompt_following, device=device, dtype=torch.float32).unsqueeze(0)
                )
                consistency_rewards.append(torch.tensor(consistency, device=device, dtype=torch.float32).unsqueeze(0))
                perceptual_quality_rewards.append(
                    torch.tensor(perceptual, device=device, dtype=torch.float32).unsqueeze(0)
                )

        _all_latents = torch.cat(_all_latents, dim=0)
        _all_log_probs = torch.cat(_all_log_probs, dim=0)
        _all_rewards = torch.cat(_all_rewards, dim=0)

        all_latents.append(_all_latents)
        all_log_probs.append(_all_log_probs)
        all_rewards.append(_all_rewards)

    # Concatenate per-component rewards if collected; otherwise set to None
    prompt_following_rewards = torch.cat(prompt_following_rewards, dim=0) if len(prompt_following_rewards) > 0 else None
    consistency_rewards = torch.cat(consistency_rewards, dim=0) if len(consistency_rewards) > 0 else None
    perceptual_quality_rewards = (
        torch.cat(perceptual_quality_rewards, dim=0) if len(perceptual_quality_rewards) > 0 else None
    )

    return (
        all_rewards,
        all_latents,
        all_log_probs,
        sigma_schedule,
        prompt_following_rewards,
        consistency_rewards,
        perceptual_quality_rewards,
        None,
    )


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def train_one_step(
    args,
    device,
    transformer,
    vae,
    tokenizer,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    max_grad_norm,
    preprocess_val,
    ema_handler,
    images_dir,
    transformer_ref,
):
    total_loss = 0.0
    total_policy_loss = 0.0
    total_kl_loss = 0.0
    adv_clip_max = args.adv_clip_max
    rank = int(os.environ.get("RANK", 0))
    optimizer.zero_grad()
    # Ensure grad_norm is always defined
    grad_norm_val = torch.tensor(0.0, device=device, dtype=torch.float32)
    batch = next(loader)
    negative_text_hidden_states = None
    negative_text_attention_masks = None
    if isinstance(batch, (list, tuple)):
        # Support both no-CFG (3-tuple) and CFG (6-tuple) collate outputs
        if len(batch) == 7:
            (
                encoder_hidden_states,
                text_attention_masks,
                ref_latents,
                negative_text_hidden_states,
                negative_text_attention_masks,
                edit_prompts,
                ref_images,
            ) = batch
        elif len(batch) >= 4:
            encoder_hidden_states, text_attention_masks, ref_latents, edit_prompts, ref_images = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch[4],
            )
        else:
            raise ValueError("Dataloader must return 3 or 6 items depending on CFG mode")
    elif isinstance(batch, dict):
        encoder_hidden_states = batch.get("prompt_embed") or batch.get("encoder_hidden_states")
        text_attention_masks = batch.get("prompt_attention_mask") or batch.get("text_attention_mask")
        ref_latents = batch.get("ref_latents")
        edit_prompts = batch.get("edit_prompts")
        # Optional negatives for CFG
        negative_text_hidden_states = batch.get("negative_prompt_embed") or batch.get("negative_encoder_hidden_states")
        negative_text_attention_masks = batch.get("negative_prompt_attention_mask") or batch.get(
            "negative_text_attention_mask"
        )
        if encoder_hidden_states is None or text_attention_masks is None or edit_prompts is None:
            raise ValueError(
                "Dataloader dict must include prompt_embed/encoder_hidden_states, "
                "prompt_attention_mask/text_attention_mask, caption/captions"
            )
    else:
        # Assume a simple 3-tuple like object
        try:
            encoder_hidden_states, text_attention_masks, ref_latents, edit_prompts = batch
        except Exception as e:
            raise ValueError(f"Unsupported dataloader batch format: {type(batch)}") from e

    if args.use_group:

        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = [
            repeat_tensor(_encoder_hidden_states) for _encoder_hidden_states in encoder_hidden_states
        ]
        text_attention_masks = [repeat_tensor(_text_attention_masks) for _text_attention_masks in text_attention_masks]
        if negative_text_hidden_states is not None:
            negative_text_hidden_states = [
                repeat_tensor(_negative_text_hidden_states)
                for _negative_text_hidden_states in negative_text_hidden_states
            ]
        else:
            negative_text_hidden_states = [None] * len(encoder_hidden_states)
        if negative_text_attention_masks is not None:
            negative_text_attention_masks = [
                repeat_tensor(_negative_text_attention_masks)
                for _negative_text_attention_masks in negative_text_attention_masks
            ]
        else:
            negative_text_attention_masks = [None] * len(encoder_hidden_states)
        ref_latents = [inner_list * args.num_generations for inner_list in ref_latents]

    # text_attention_mask 已由数据集提供
    (
        rewards,
        all_latents,
        all_log_probs,
        sigma_schedule,
        prompt_following_rewards,
        consistency_rewards,
        perceptual_quality_rewards,
        _,
    ) = sample_reference_model(
        args,
        device,
        transformer,
        vae,
        encoder_hidden_states,
        text_attention_masks,
        ref_latents,
        tokenizer,
        edit_prompts,
        preprocess_val,
        images_dir,
        args.sample_batch_size,
        ref_images=ref_images,
        negative_text_hidden_states=negative_text_hidden_states,
        negative_text_attention_masks=negative_text_attention_masks,
    )

    all_gathered_reward = []
    for (
        reward,
        latents,
        log_probs,
        encoder_hidden_state,
        text_attention_mask,
        negative_text_hidden_state,
        negative_text_attention_mask,
        ref_latent,
    ) in zip(
        rewards,
        all_latents,
        all_log_probs,
        encoder_hidden_states,
        text_attention_masks,
        negative_text_hidden_states,
        negative_text_attention_masks,
        ref_latents,
        strict=True,
    ):
        batch_size = latents.shape[0]

        gathered_reward = gather_tensor(reward)
        all_gathered_reward.append(gathered_reward)
        if args.use_group:
            all_advantages = torch.zeros_like(reward)
            group_mean = reward.mean()
            group_std = reward.std() + 1e-8
            if group_mean < args.reward_threshold:
                all_advantages[:] = 0
            else:
                all_advantages[:] = (reward - group_mean) / group_std
        else:
            all_advantages = (reward - gathered_reward.mean()) / (gathered_reward.std() + 1e-8)

        timestep_values = [int((t.item() if torch.is_tensor(t) else float(t)) * 1000) for t in sigma_schedule][
            : args.sampling_steps
        ]
        device = latents.device
        timesteps = torch.tensor(timestep_values, device=latents.device, dtype=torch.long)

        train_timesteps = random.sample(range(len(timestep_values)), int(len(timestep_values) * args.timestep_fraction))
        grpo_batch_size = args.grpo_batch_size

        for i, timestep_index in enumerate(train_timesteps):
            num_chunks = max(1, (batch_size + grpo_batch_size - 1) // grpo_batch_size)
            batch_indices = torch.chunk(torch.arange(batch_size), num_chunks)
            start_time = time.time()
            for index, batch_idx in enumerate(batch_indices):
                batch_idx_list = batch_idx.tolist()
                latents_chunk = latents[batch_idx]
                log_probs_chunk = log_probs[batch_idx]
                encoder_hidden_state_chunk = encoder_hidden_state[batch_idx]
                text_attention_mask_chunk = text_attention_mask[batch_idx]
                negative_text_hidden_state_chunk = (
                    negative_text_hidden_state[batch_idx] if negative_text_hidden_state is not None else None
                )
                negative_text_attention_mask_chunk = (
                    negative_text_attention_mask[batch_idx] if negative_text_attention_mask is not None else None
                )
                ref_latent_chunk = [ref_latent[_index] for _index in batch_idx_list]

                current_latents = latents_chunk[:, timestep_index]
                next_latents = latents_chunk[:, timestep_index + 1]

                new_log_probs, prev_sample_mean, std_dev_t = grpo_one_step(
                    args,
                    current_latents,
                    next_latents,
                    encoder_hidden_state_chunk,
                    text_attention_mask_chunk,
                    transformer,
                    timesteps[timestep_index : timestep_index + 1],
                    sigma_schedule,  # continuous extended timesteps returned from sampler
                    timestep_index,
                    ref_latent_chunk,
                    negative_text_hidden_states=negative_text_hidden_state_chunk,
                    negative_text_attention_mask=negative_text_attention_mask_chunk,
                )

                if transformer_ref is not None:
                    with torch.no_grad():
                        new_log_probs_ref, prev_sample_mean_ref, std_dev_t_ref = grpo_one_step(
                            args,
                            current_latents,
                            next_latents,
                            encoder_hidden_state_chunk,
                            text_attention_mask_chunk,
                            transformer,
                            timesteps[timestep_index : timestep_index + 1],
                            sigma_schedule,  # continuous extended timesteps returned from sampler
                            timestep_index,
                            ref_latent_chunk,
                            negative_text_hidden_states=negative_text_hidden_state_chunk,
                            negative_text_attention_mask=negative_text_attention_mask_chunk,
                        )

                advantages = torch.clamp(
                    all_advantages[batch_idx],
                    -adv_clip_max,
                    adv_clip_max,
                )

                ratio = torch.exp(new_log_probs - log_probs_chunk[:, timestep_index])
                unclipped_loss = -advantages * ratio

                policy_loss = torch.mean(unclipped_loss) / (len(rewards) * len(batch_indices) * len(train_timesteps))

                if transformer_ref is not None:
                    kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1, 2), keepdim=True) / (
                        2 * std_dev_t**2
                    )
                    kl_loss = torch.mean(kl_loss)
                    loss = policy_loss + args.kl_beta * kl_loss
                else:
                    loss = policy_loss

                loss.backward()
                avg_loss = loss.detach().clone()
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                total_loss += avg_loss.item()

                avg_policy_loss = policy_loss.detach().clone()
                dist.all_reduce(avg_policy_loss, op=dist.ReduceOp.AVG)
                total_policy_loss += avg_policy_loss.item()

                if transformer_ref is not None:
                    avg_kl_loss = kl_loss.detach().clone()
                    dist.all_reduce(avg_kl_loss, op=dist.ReduceOp.AVG)
                    total_kl_loss += avg_kl_loss.item()
            end_time = time.time()
            if rank == 0:
                print(
                    f"Step {i + 1}/{len(train_timesteps)}: "
                    f"Loss {total_loss:.4f}, "
                    f"Policy Loss {total_policy_loss:.4f}, "
                    f"total_kl_loss {total_kl_loss:.4f}, "
                    f"Time {end_time - start_time:.4f}"
                )

    grad_norm_val = transformer.clip_grad_norm_(max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    dist.barrier()

    if args.editscore_reward and prompt_following_rewards is not None:
        try:
            gathered_prompt_following_rewards = gather_tensor(prompt_following_rewards)
            gathered_consistency_rewards = gather_tensor(consistency_rewards)
            gathered_perceptual_quality_rewards = gather_tensor(perceptual_quality_rewards)
        except Exception:
            gathered_prompt_following_rewards = None
            gathered_consistency_rewards = None
            gathered_perceptual_quality_rewards = None
    reward_stats = {
        "reward_mean": gathered_reward.mean().item(),
        "reward_std": gathered_reward.std().item(),
        "reward_min": gathered_reward.min().item(),
        "reward_max": gathered_reward.max().item(),
    }
    if args.editscore_reward and gathered_prompt_following_rewards is not None:
        reward_stats.update(
            {
                "prompt_following_mean": gathered_prompt_following_rewards.mean().item(),
                "prompt_following_std": gathered_prompt_following_rewards.std().item(),
                "prompt_following_min": gathered_prompt_following_rewards.min().item(),
                "prompt_following_max": gathered_prompt_following_rewards.max().item(),
            }
        )
    if args.editscore_reward and gathered_consistency_rewards is not None:
        reward_stats.update(
            {
                "consistency_mean": gathered_consistency_rewards.mean().item(),
                "consistency_std": gathered_consistency_rewards.std().item(),
                "consistency_min": gathered_consistency_rewards.min().item(),
                "consistency_max": gathered_consistency_rewards.max().item(),
            }
        )
    if args.editscore_reward and gathered_perceptual_quality_rewards is not None:
        reward_stats.update(
            {
                "perceptual_quality_mean": gathered_perceptual_quality_rewards.mean().item(),
                "perceptual_quality_std": gathered_perceptual_quality_rewards.std().item(),
                "perceptual_quality_min": gathered_perceptual_quality_rewards.min().item(),
                "perceptual_quality_max": gathered_perceptual_quality_rewards.max().item(),
            }
        )

    return total_loss, grad_norm_val.item(), reward_stats, total_policy_loss, total_kl_loss


def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group()
    torch.npu.set_device(local_rank)
    device = torch.npu.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    if args.seed is not None:
        set_seed(args.seed + rank)
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Hybrid reward only arguments
    assert args.use_hybrid_reward, "This script only supports --use_hybrid_reward mode."
    preprocess_val = None
    main_print("--> Reward model initialized: EditScoreRewardModel")
    main_print(f"--> loading Mammoth2 model from {args.pretrained_model_name_or_path}")

    # Load transformer in Mammoth style (aligned with embedding preprocessing)
    conf_path = None
    if hasattr(args, "unified_model_config") and args.unified_model_config:
        conf_path = args.unified_model_config
    elif args.pretrained_model_name_or_path:
        if os.path.isdir(args.pretrained_model_name_or_path):
            conf_path = os.path.join(args.pretrained_model_name_or_path, "config.yml")
        else:
            conf_path = args.pretrained_model_name_or_path
    if conf_path is None or not os.path.exists(conf_path):
        raise ValueError(f"Cannot find Mammoth2 config yaml, got: {conf_path}")
    model_args = OmegaConf.load(conf_path)

    arch_opt = getattr(model_args.model, "arch_opt", {})
    transformer = Transformer2DModel(
        patch_size=arch_opt.get("patch_size", 2),
        in_channels=arch_opt.get("in_channels", 16),
        out_channels=arch_opt.get("out_channels", None),
        hidden_size=arch_opt.get("hidden_size", 2304),
        num_layers=arch_opt.get("num_layers", 26),
        num_refiner_layers=arch_opt.get("num_refiner_layers", 2),
        num_attention_heads=arch_opt.get("num_attention_heads", 24),
        num_kv_heads=arch_opt.get("num_kv_heads", 8),
        multiple_of=arch_opt.get("multiple_of", 256),
        ffn_dim_multiplier=arch_opt.get("ffn_dim_multiplier", None),
        norm_eps=arch_opt.get("norm_eps", 1e-5),
        axes_dim_rope=tuple(arch_opt.get("axes_dim_rope", (32, 32, 32))),
        axes_lens=tuple(arch_opt.get("axes_lens", (300, 512, 512))),
        text_feat_dim=arch_opt.get("text_feat_dim", 1024),
        timestep_scale=arch_opt.get("timestep_scale", 1.0),
        is_image_embedder=arch_opt.get("is_image_embedder", False),
        enable_cross_attention=arch_opt.get("enable_cross_attention", False),
    )

    if args.kl_beta > 0:
        transformer_ref = Transformer2DModel(
            patch_size=arch_opt.get("patch_size", 2),
            in_channels=arch_opt.get("in_channels", 16),
            out_channels=arch_opt.get("out_channels", None),
            hidden_size=arch_opt.get("hidden_size", 2304),
            num_layers=arch_opt.get("num_layers", 26),
            num_refiner_layers=arch_opt.get("num_refiner_layers", 2),
            num_attention_heads=arch_opt.get("num_attention_heads", 24),
            num_kv_heads=arch_opt.get("num_kv_heads", 8),
            multiple_of=arch_opt.get("multiple_of", 256),
            ffn_dim_multiplier=arch_opt.get("ffn_dim_multiplier", None),
            norm_eps=arch_opt.get("norm_eps", 1e-5),
            axes_dim_rope=tuple(arch_opt.get("axes_dim_rope", (32, 32, 32))),
            axes_lens=tuple(arch_opt.get("axes_lens", (300, 512, 512))),
            text_feat_dim=arch_opt.get("text_feat_dim", 1024),
            timestep_scale=arch_opt.get("timestep_scale", 1.0),
            is_image_embedder=arch_opt.get("is_image_embedder", False),
            enable_cross_attention=arch_opt.get("enable_cross_attention", False),
        )
    else:
        transformer_ref = None
    try:
        replace_rmsnorm_with_custom(transformer)
        if transformer_ref is not None:
            replace_rmsnorm_with_custom(transformer_ref)
    except Exception:
        pass

    # Load transformer weights from unified model checkpoint
    ckpt_path = getattr(args, "unified_model_path", None)
    if ckpt_path is None and os.path.isdir(args.pretrained_model_name_or_path or ""):
        for name in [
            "pytorch_model_fsdp.bin",
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.safetensors",
        ]:
            candidate = os.path.join(args.pretrained_model_name_or_path, name)
            if os.path.exists(candidate):
                ckpt_path = candidate
                break
    if ckpt_path is not None and os.path.exists(ckpt_path):
        try:
            from safetensors.torch import load_file as safe_load_file

            state = (
                safe_load_file(ckpt_path)
                if ckpt_path.endswith(".safetensors")
                else torch.load(ckpt_path, map_location="cpu")
            )
        except Exception:
            state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        transformer_state = {k[len("transformer.") :]: v for k, v in state.items() if k.startswith("transformer.")}
        missing_keys, unexpected_keys = transformer.load_state_dict(transformer_state, strict=False)
        main_print(f"--> Loaded transformer weights from unified checkpoint: {ckpt_path}")
        if len(missing_keys) > 0:
            main_print(f"--> Missing transformer keys: {len(missing_keys)}")
        if len(unexpected_keys) > 0:
            main_print(f"--> Unexpected transformer keys: {len(unexpected_keys)}")

        if transformer_ref is not None:
            missing_keys_ref, unexpected_keys_ref = transformer_ref.load_state_dict(transformer_state, strict=False)
            if len(missing_keys_ref) > 0:
                main_print(f"--> Missing transformer_ref keys: {len(missing_keys_ref)}")
            if len(unexpected_keys_ref) > 0:
                main_print(f"--> Unexpected transformer_ref keys: {len(unexpected_keys_ref)}")
    else:
        main_print(
            "--> Warning: unified_model_path not provided/found; using randomly initialized transformer per config"
        )

    # Ensure gradient checkpointing hook is installed inside the model before FSDP wrapping
    if args.gradient_checkpointing:
        try:
            transformer.enable_gradient_checkpointing()
        except Exception:
            pass

    # Diagnostics: print transformer config and lightweight param stats (rank 0)
    try:
        if dist.get_rank() == 0:
            main_print(
                f"--> Transformer config: "
                f"hidden_size={transformer.config.hidden_size}, "
                f"layers={transformer.config.num_layers}, "
                f"heads={transformer.config.num_attention_heads}, "
                f"text_feat_dim={transformer.config.text_feat_dim}"
            )
            main_print(
                f"--> RoPE axes_dim={transformer.config.axes_dim_rope}, axes_lens={transformer.config.axes_lens}"
            )
            trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
            first_param = next(transformer.parameters())
            main_print(
                f"--> Trainable params: {trainable_params / 1e6:.2f}M, "
                f"dtype={first_param.dtype}, "
                f"abs-mean={first_param.detach().abs().mean().item():.6f}"
            )
    except Exception:
        pass
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    transformer = FSDP(
        transformer,
        **fsdp_kwargs,
    )
    if transformer_ref is not None:
        transformer_ref = FSDP(
            transformer_ref,
            **fsdp_kwargs,
        )
    ema_handler = None
    if args.use_ema:
        ema_handler = FSDP_EMA(transformer, args.ema_decay, rank)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer, no_split_modules, args.selective_checkpointing)

    vae = AutoencoderKL.from_pretrained(
        args.vae_model_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    try:
        if dist.get_rank() == 0:
            vae_in = getattr(vae.config, "in_channels", None)
            vae_sf = getattr(vae, "scaling_factor", getattr(vae.config, "scaling_factor", None))
            vae_shift = getattr(vae, "shift_factor", getattr(vae.config, "shift_factor", None))
            main_print(f"--> VAE: in_channels={vae_in}, scaling_factor={vae_sf}, shift_factor={vae_shift}")
    except Exception:
        pass
    main_print(f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}")
    transformer.train()
    if transformer_ref is not None:
        transformer_ref.eval()
    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    init_steps = 0
    main_print(f"optimizer: {optimizer}")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentEditDataset(
        args.data_json_path,
        use_negative=bool(getattr(args, "use_negative_cfg", 1)),
    )
    sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    project = "mammoth2.0_exp_npu"

    def generate_experiment_name(args, world_size):
        name_parts = []
        name_parts.append("m2")
        name_parts.append(f"lr{args.learning_rate:.0e}")
        name_parts.append(f"bs{args.train_batch_size}")
        name_parts.append(f"ga{args.gradient_accumulation_steps}")
        name_parts.append(f"g{args.num_generations}")
        name_parts.append(f"s{args.sampling_steps}")
        name_parts.append(f"e{args.eta}")
        name_parts.append(f"gpu{world_size}")
        if args.use_group:
            name_parts.append("grp")
        if args.use_ema:
            name_parts.append("ema")
        if args.init_same_noise:
            name_parts.append("sn")
        if args.h and args.w:
            name_parts.append(f"{args.h}x{args.w}")
        return "_".join(name_parts)

    experiment_name = generate_experiment_name(args, world_size)
    main_print(f"--> Experiment name: {experiment_name}")
    if args.output_dir is not None:
        original_output_dir = args.output_dir
        args.output_dir = os.path.join(original_output_dir, experiment_name)
        main_print(f"--> Original output_dir: {original_output_dir}")
        main_print(f"--> Modified output_dir: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

    if dist.get_rank() == 0:
        wandb_config = {
            "model": "Mammoth2",
            "reward_model": "hybrid_reward",
            "learning_rate": args.learning_rate,
            "batch_size": args.train_batch_size,
            "num_generations": args.num_generations,
            "sampling_steps": args.sampling_steps,
            "eta": args.eta,
            "use_group": args.use_group,
            "use_ema": args.use_ema,
            "init_same_noise": args.init_same_noise,
            "image_size": f"{args.h}x{args.w}" if args.h and args.w else "unknown",
            "shift": args.shift,
            "timestep_fraction": args.timestep_fraction,
            "clip_range": args.clip_range,
            "adv_clip_max": args.adv_clip_max,
            "max_train_steps": args.max_train_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "original_output_dir": original_output_dir if args.output_dir is not None else None,
            "final_output_dir": args.output_dir,
        }
        try:
            wandb.init(
                project=project,
                name=experiment_name,
                config=wandb_config,
            )
        except Exception as e:
            print(f"Error initializing wandb: {e}")

    total_batch_size = (
        args.train_batch_size * world_size * args.gradient_accumulation_steps / args.sp_size * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}")
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = "
        f"{sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        disable=local_rank > 0,
    )
    loader = sp_parallel_dataloader_wrapper_imgedit(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    images_dir = os.path.join(args.output_dir, "generated_images")
    os.makedirs(images_dir, exist_ok=True)

    step_times = deque(maxlen=100)
    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        for step in range(init_steps + 1, args.max_train_steps + 1):
            start_time = time.time()
            if step % args.checkpointing_steps == 0:
                save_checkpoint(transformer, rank, args.output_dir, step, epoch)
                if args.use_ema and ema_handler:
                    save_ema_checkpoint(ema_handler, rank, args.output_dir, step, epoch, dict(transformer.config))
                dist.barrier()

            loss, grad_norm, reward_stats, policy_loss, kl_loss = train_one_step(
                args,
                device,
                transformer,
                vae,
                None,
                optimizer,
                lr_scheduler,
                loader,
                None,
                args.max_grad_norm,
                preprocess_val,
                ema_handler,
                images_dir,
                transformer_ref,
            )

            if args.dry_run_sde:
                print("[DryRun-SDE] Completed first training step with SDE sampling. Exiting.")
                return

            if args.use_ema and ema_handler:
                ema_handler.update(transformer)
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                    "policy_loss": f"{policy_loss:.4f}",
                    "kl_loss": f"{kl_loss:.4f}",
                }
            )
            progress_bar.update(1)
            if dist.get_rank() == 0:
                log_data = {
                    "train_loss": loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": grad_norm,
                    "reward_mean": reward_stats["reward_mean"],
                    "reward_std": reward_stats["reward_std"],
                    "reward_min": reward_stats["reward_min"],
                    "reward_max": reward_stats["reward_max"],
                    "policy_loss": policy_loss,
                    "kl_loss": kl_loss,
                }
                # If hybrid reward components exist, include their stats (align with train_grpo_flux_hybrid)
                if args.editscore_reward and "prompt_following_mean" in reward_stats:
                    log_data.update(
                        {
                            "prompt_following_mean": reward_stats["prompt_following_mean"],
                            "prompt_following_std": reward_stats["prompt_following_std"],
                            "prompt_following_min": reward_stats["prompt_following_min"],
                            "prompt_following_max": reward_stats["prompt_following_max"],
                        }
                    )
                if args.editscore_reward and "consistency_mean" in reward_stats:
                    log_data.update(
                        {
                            "consistency_mean": reward_stats["consistency_mean"],
                            "consistency_std": reward_stats["consistency_std"],
                            "consistency_min": reward_stats["consistency_min"],
                            "consistency_max": reward_stats["consistency_max"],
                        }
                    )
                if args.editscore_reward and "perceptual_quality_mean" in reward_stats:
                    log_data.update(
                        {
                            "perceptual_quality_mean": reward_stats["perceptual_quality_mean"],
                            "perceptual_quality_std": reward_stats["perceptual_quality_std"],
                            "perceptual_quality_min": reward_stats["perceptual_quality_min"],
                            "perceptual_quality_max": reward_stats["perceptual_quality_max"],
                        }
                    )
                wandb.log(log_data, step=step)

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--unified_model_config", type=str, default=None, help="Path to Mammoth unified config.yml")
    parser.add_argument(
        "--unified_model_path", type=str, default=None, help="Path to unified model weights (.safetensors/.bin)"
    )
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    # CFG options (align with sampling script)
    parser.add_argument(
        "--use_negative_cfg",
        action="store_true",
        default=False,
        help="Enable CFG using negative embeddings/masks from dataset",
    )
    parser.add_argument("--cfg_rate", type=float, default=0.0, help="CFG rate in dataset")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale used when --use_negative_cfg is on")
    parser.add_argument(
        "--cfg_scale_image", type=float, default=2.0, help="CFG scale used when --use_negative_cfg is on for image"
    )

    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )
    parser.add_argument("--fsdp_sharding_startegy", default="full")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to apply.")
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    parser.add_argument(
        "--h",
        type=int,
        default=None,
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether compute advantages for each prompt",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--use_hpsv2",
        action="store_true",
        default=False,
        help="whether use hpsv2 as reward model",
    )
    parser.add_argument(
        "--use_hpsv3",
        action="store_true",
        default=False,
        help="whether use hpsv3 as reward model",
    )
    parser.add_argument(
        "--use_pickscore",
        action="store_true",
        default=False,
        help="whether use pickscore as reward model",
    )
    parser.add_argument(
        "--use_ocr",
        action="store_true",
        default=False,
        help="whether use OCR as reward model for text rendering quality evaluation",
    )
    parser.add_argument(
        "--ocr_lang",
        type=str,
        default="en",
        help="language for OCR recognition (default: en)",
    )
    parser.add_argument(
        "--ocr_server_url",
        type=str,
        default="http://localhost:5000",
        help="URL of the OCR server (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--use_ocr_client",
        action="store_true",
        default=False,
        help="Whether to use an OCR client instead of a local model.",
    )
    parser.add_argument(
        "--use_inhouse_ocr",
        action="store_true",
        default=False,
        help="Whether to use inhouse OCR service as reward model.",
    )
    parser.add_argument(
        "--ocr_threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for OCR results (default: 0.8)",
    )
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_fraction",
        type=float,
        default=1.0,
        help="timestep downsample ratio",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=1e-4,
        help="clip range for grpo",
    )
    parser.add_argument(
        "--adv_clip_max",
        type=float,
        default=5.0,
        help="clipping advantage",
    )
    parser.add_argument("--use_ema", action="store_true", help="Enable Exponential Moving Average of model weights.")
    parser.add_argument(
        "--show_progress_bar",
        action="store_true",
        default=False,
        help="Whether to show progress bar during sampling",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=1,
        help="Batch size for sampling during training",
    )
    parser.add_argument(
        "--grpo_batch_size",
        type=int,
        default=1,
        help="Batch size for sampling during training",
    )
    parser.add_argument(
        "--dry_run_sde",
        action="store_true",
        default=False,
        help="Run a single training step using SDE sampling only to dump images then exit",
    )
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=0.0,
        help="If upper than 0, skip sample reward smaller than this value",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=0.0,
        help="KL divergence beta for GRPO sampling",
    )
    # epsilon(t) scheduling for GRPO sampling
    parser.add_argument(
        "--eps_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine", "poly"],
        help="epsilon(t) schedule; front-big-later-small preferred",
    )
    parser.add_argument("--eps_power", type=float, default=2.0, help="power for poly schedule: eps(t)=eta*(1-t)^power")
    parser.add_argument(
        "--eps_min_ratio", type=float, default=0.0, help="lower bound for epsilon scale to prevent vanishing noise"
    )
    # Hybrid reward only arguments
    parser.add_argument("--use_hybrid_reward", action="store_true", default=True)
    parser.add_argument(
        "--hpsv3_checkpoint_path", type=str, default="/mnt/bn/seutao-hl/DATA/ckpts/MizzenAI/HPSv3/HPSv3.safetensors"
    )
    parser.add_argument("--hpsv3_weight", type=float, default=0.5)
    parser.add_argument("--ocr_weight", type=float, default=0.5)
    parser.add_argument("--doubao_weight", type=float, default=0.0)
    parser.add_argument("--normalize_rewards", action="store_true", default=True)
    parser.add_argument("--enable_reward_logging", action="store_true", default=True)
    parser.add_argument("--doubao_model_name", type=str, default="ep-20250715204328-66tgc")
    parser.add_argument("--doubao_base_url", type=str, default="https://ark-cn-beijing.bytedance.net/api/v3")
    parser.add_argument("--doubao_max_retries", type=int, default=3)
    parser.add_argument("--doubao_retry_delay", type=float, default=1.0)
    parser.add_argument("--doubao_rate_limit_delay", type=float, default=2.0)
    parser.add_argument("--doubao_max_wait_time", type=float, default=300.0)
    parser.add_argument("--doubao_verbose_logging", action="store_true", default=False)
    parser.add_argument("--doubao_batch_size", type=int, default=None)
    parser.add_argument("--editscore_reward", action="store_true", default=False)
    parser.add_argument("--editscore_psm", type=str, default="bes.general_audit.editscore_7b")
    parser.add_argument("--editscore_num_pass", type=int, default=5)
    parser.add_argument("--reward_num_workers", type=int, default=1)

    args = parser.parse_args()
    main(args)
