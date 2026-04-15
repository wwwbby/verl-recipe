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
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import logging
import math
import os
import re
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from verl import DataProto
from verl.utils.device import get_device_name

logger = logging.getLogger(__name__)

__all__ = ["HFRollout"]


class GRPOMockScheduler:
    """
    A mock scheduler to inject GRPO SDE math into the built-in `decode_diffusion_image`
    while recording the trajectories.

    The timestep schedule matches the native FlowMatchEulerDiscreteScheduler with
    dynamic_time_shift=True: t' = t / (m + t*(1-m)) where m = sqrt(num_tokens) / 40.
    This concentrates steps near t=0 (low-noise end) at higher resolutions, exactly as
    done during standard inference, keeping rollout and training schedules consistent.
    """

    def __init__(self, sample_steps=10, eta=0.01, num_tokens=None):
        self.sample_steps = sample_steps
        self.eta = eta

        timesteps = torch.linspace(0, 1, sample_steps + 1)
        if num_tokens is not None:
            # Apply the same resolution-dependent warp as FlowMatchEulerDiscreteScheduler.
            # Equivalent to omni_time_shift(m, t); endpoints 0 and 1 are preserved.
            m = (num_tokens**0.5) / 40.0
            timesteps = timesteps / (m + timesteps * (1.0 - m))
        self.timesteps = timesteps

        self.all_latents = []
        self.all_log_probs = []
        self.current_step_idx = 0

    def set_timesteps(self, num_inference_steps, device):
        self.timesteps = self.timesteps.to(device)
        return self.timesteps[:-1]  # return all but last for loop

    def compute_dance_grpo_step(
        self,
        latents: torch.Tensor,
        model_output: torch.Tensor,
        t: float,
        t_next: float,
        eta: float,
        prev_sample: torch.Tensor = None,
        sde_solver: bool = True,
    ):
        dt = t_next - t

        # Cast to float32 to match the precision used in update_actor, ensuring
        # that old_log_probs (recorded here) and new_log_probs (recomputed during
        # training) are numerically consistent when the model weights are unchanged.
        # Without this, BF16 rounding in prev_sample_mean causes ratio != 1 at step 1.
        latents = latents.to(torch.float32)
        model_output = model_output.to(torch.float32)

        x_hat = latents + (1.0 - t) * model_output

        prev_sample_mean = latents + dt * model_output

        score_estimate = -(latents - t * x_hat) / ((1.0 - t) ** 2 + 1e-12)

        std_dev_t = eta * torch.sqrt(torch.abs(torch.tensor(dt, device=latents.device, dtype=torch.float32)) + 1e-12)

        if sde_solver:
            prev_sample_mean = prev_sample_mean + (0.5 * (eta**2)) * score_estimate * dt

        if prev_sample is None:
            noise = torch.randn_like(prev_sample_mean)
            prev_sample = prev_sample_mean + noise * std_dev_t

        diff = prev_sample.to(torch.float32) - prev_sample_mean
        std32 = std_dev_t
        var_t = (std32**2).clamp_min(1e-20)
        log_std = torch.log(std32.clamp_min(1e-20))
        log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=latents.device, dtype=torch.float32))

        log_prob = (-(diff**2) / (2 * var_t)) - log_std - 0.5 * log_two_pi
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return prev_sample, log_prob

    def step(self, model_output, timestep, sample, return_dict=False):
        # Record input latents on the very first step
        if self.current_step_idx == 0:
            self.all_latents.append(sample.clone())

        t = self.timesteps[self.current_step_idx].item()
        t_next = self.timesteps[self.current_step_idx + 1].item()

        prev_sample, log_prob = self.compute_dance_grpo_step(
            latents=sample, model_output=model_output, t=t, t_next=t_next, eta=self.eta, prev_sample=None
        )

        self.all_latents.append(prev_sample.clone())
        self.all_log_probs.append(log_prob)
        self.current_step_idx += 1

        return (prev_sample,)


class HFRollout:
    def __init__(
        self, mllm_model: nn.Module, dit_model: nn.Module, vae_model: nn.Module, full_model, processor, config
    ):
        self.config = config
        self.mllm_model = mllm_model
        self.dit_model = dit_model
        self.vae_model = vae_model
        self.full_model = full_model
        self.processor = processor

    @staticmethod
    def _pad_or_truncate_conditioning(hidden_states: torch.Tensor, attention_mask: torch.Tensor, max_seq_len: int):
        """Pad/truncate [B, L, D] and [B, L] tensors to a fixed sequence length."""
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states should be 3D [B, L, D], got {tuple(hidden_states.shape)}")
        if attention_mask.dim() != 2:
            raise ValueError(f"attention_mask should be 2D [B, L], got {tuple(attention_mask.shape)}")

        cur_len = hidden_states.shape[1]
        if cur_len == max_seq_len:
            return hidden_states.contiguous(), attention_mask.contiguous()

        if cur_len > max_seq_len:
            # Keep the right-most tokens to match left-padding behavior in processor inputs.
            return hidden_states[:, -max_seq_len:, :].contiguous(), attention_mask[:, -max_seq_len:].contiguous()

        pad_len = max_seq_len - cur_len
        hs_pad = torch.zeros(
            hidden_states.shape[0],
            pad_len,
            hidden_states.shape[2],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        am_pad = torch.zeros(
            attention_mask.shape[0],
            pad_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        hidden_states = torch.cat([hs_pad, hidden_states], dim=1)
        attention_mask = torch.cat([am_pad, attention_mask], dim=1)
        return hidden_states.contiguous(), attention_mask.contiguous()

    @torch.no_grad()
    def _generate_minibatch(self, data: DataProto) -> DataProto:
        device = get_device_name()
        rollout_cfg = self.config
        h = int(getattr(rollout_cfg, "height", 512))
        w = int(getattr(rollout_cfg, "width", 512))
        vae_scale_factor = int(getattr(rollout_cfg, "vae_scale_factor", 16))
        num_inference_steps = int(getattr(rollout_cfg, "num_inference_steps", 40))
        eta = float(getattr(rollout_cfg, "eta", 0.3))
        conditioning_max_length = int(getattr(rollout_cfg, "prompt_length", 512))
        init_same_noise = bool(getattr(rollout_cfg, "init_same_noise", True))
        output_dir = str(getattr(rollout_cfg, "output_dir", "logs"))
        cfg_scale = float(getattr(rollout_cfg, "cfg_scale", 7.0))
        text_cfg_scale = float(getattr(rollout_cfg, "text_cfg_scale", 9.0))

        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        os.makedirs(output_dir, exist_ok=True)

        prompts = data.non_tensor_batch["prompts"]
        _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if _rank == 0:
            logger.info(f"Starting DanceGRPO Rollout with batch size: {len(prompts)}")
        rollout_n = len(prompts)

        # 1. Dynamically process numpy prompts into message list format
        from recipe.dance_grpo.mammothmoda2.model import DEFAULT_NEGATIVE_PROMPT
        from recipe.dance_grpo.mammothmoda2.utils import decode_diffusion_image

        # Process prompts dynamically into chat templates
        main_prompt = prompts[0]
        messages = [{"role": "user", "content": [{"type": "text", "text": str(main_prompt)}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = None, None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            num_images_per_prompt=1,
            cfg_scale=cfg_scale,  # built-in CFG scaling
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)

        # 1. Run Built-in MLLM generate
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            generated_ids, attention_mask = self.full_model.generate(**inputs, cfg_scale=cfg_scale)

        generated_ids = torch.repeat_interleave(generated_ids, repeats=rollout_n, dim=0)
        attention_mask = torch.repeat_interleave(attention_mask, repeats=rollout_n, dim=0)
        for key, val in inputs.items():
            if val is not None and isinstance(val, torch.Tensor):
                inputs[key] = torch.repeat_interleave(val, repeats=rollout_n, dim=0)

        # 2. Decode Images while injecting GRPO Scheduler and capturing text conditioning
        # num_tokens matches native decode_diffusion_image latent shape: (2*h/vae_scale_factor) * (2*w/vae_scale_factor)
        latent_h = 2 * h // vae_scale_factor
        latent_w = 2 * w // vae_scale_factor
        num_tokens = latent_h * latent_w
        grpo_scheduler = GRPOMockScheduler(sample_steps=num_inference_steps, eta=eta, num_tokens=num_tokens)

        # Capture the text conditioning tensors produced by encode_full_prompts so they can
        # be reused during the training step (MLLM is frozen, no need to re-run it).
        captured_conditioning = {}
        from recipe.dance_grpo.mammothmoda2.utils import t2i_utils as _t2i_utils

        _orig_encode_full_prompts = _t2i_utils.encode_full_prompts

        def _capturing_encode_full_prompts(*args, **kwargs):
            result = _orig_encode_full_prompts(*args, **kwargs)
            (
                text_cond,
                text_cond_mask,
                image_cond,
                image_cond_mask,
                neg_cond,
                neg_mask,
            ) = result
            # Store the ORIGINAL (unpadded) conditioning.  Padding for DataProto
            # transport happens later; the rollout DiT must see the native lengths.
            captured_conditioning["text_hidden_states"] = text_cond.detach().cpu()
            captured_conditioning["text_attention_mask"] = text_cond_mask.detach().cpu()
            captured_conditioning["text_seq_len"] = text_cond.shape[1]
            if image_cond is not None:
                captured_conditioning["image_hidden_states"] = image_cond.detach().cpu()
                captured_conditioning["image_attention_mask"] = image_cond_mask.detach().cpu()
                captured_conditioning["image_seq_len"] = image_cond.shape[1]
            if neg_cond is not None:
                captured_conditioning["negative_text_hidden_states"] = neg_cond.detach().cpu()
                captured_conditioning["negative_text_attention_mask"] = neg_mask.detach().cpu()
                captured_conditioning["neg_seq_len"] = neg_cond.shape[1]
            # Return the ORIGINAL result unchanged so processing() sees native lengths.
            return result

        # We patch `FlowMatchEulerDiscreteScheduler` inside `t2i_utils` to return our mock scheduler
        # and also intercept encode_full_prompts to capture conditioning tensors.
        with (
            patch(
                "recipe.dance_grpo.mammothmoda2.utils.t2i_utils.FlowMatchEulerDiscreteScheduler",
                return_value=grpo_scheduler,
            ),
            patch(
                "recipe.dance_grpo.mammothmoda2.utils.t2i_utils.retrieve_timesteps",
                return_value=(grpo_scheduler.set_timesteps(num_inference_steps, device), num_inference_steps),
            ),
            patch(
                "recipe.dance_grpo.mammothmoda2.utils.t2i_utils.encode_full_prompts",
                side_effect=_capturing_encode_full_prompts,
            ),
        ):
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                # Note: decode_diffusion_image handles the CFG logic natively if cfg_scale > 1.0
                imgs_list = decode_diffusion_image(
                    prompt_ids=inputs.input_ids,
                    generated_ids=generated_ids,
                    attention_mask=attention_mask,
                    negative_ids=inputs.get("negative_ids", None),
                    negative_mask=inputs.get("negative_mask", None),
                    model=self.full_model,
                    tokenizer=self.processor.tokenizer,
                    output_dir=output_dir,
                    num_images_per_prompt=rollout_n,
                    text_guidance_scale=text_cfg_scale,
                    vae_scale_factor=16,
                    cfg_range=(0.0, 1.0),
                    num_inference_steps=num_inference_steps,
                    height=h,
                    width=w,
                    device=device,
                    init_same_noise=init_same_noise,
                    return_img_only=True,
                )
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                global_step = getattr(self, "_current_global_step", 0)
                experiment_name = getattr(self, "_current_experiment_name", "dance_grpo")
                step_dir = os.path.join(output_dir, experiment_name, f"step_{global_step:06d}")
                os.makedirs(step_dir, exist_ok=True)
                safe_prompt = re.sub(r"[^A-Za-z0-9._-]+", "_", str(main_prompt)).strip("_")[:32] or "prompt"
                image_paths = []
                for idx, decoded_img in enumerate(imgs_list):
                    fname = f"rank_{rank}_img_{idx}_{safe_prompt}.png"
                    fpath = os.path.join(step_dir, fname)
                    decoded_img.save(fpath)
                    image_paths.append(fpath)

        # Format Recorded trajectories
        # [B, steps+1, C, H, W]
        all_latents = torch.stack(grpo_scheduler.all_latents, dim=1)
        all_log_probs = torch.stack(grpo_scheduler.all_log_probs, dim=1)  # [B, steps]
        # [steps+1]
        sigma_schedule = grpo_scheduler.timesteps
        all_images = np.array(imgs_list)

        # Build conditioning tensors – broadcast to match rollout_n (all samples share the same prompt).
        # Conditioning is stored at its ORIGINAL length from encode_full_prompts.
        # Pad to conditioning_max_length for DataProto.concat (requires uniform shapes),
        # and store the actual sequence length so training can truncate back.
        text_hidden_states = captured_conditioning["text_hidden_states"].expand(rollout_n, -1, -1).contiguous()
        text_attention_mask = captured_conditioning["text_attention_mask"].expand(rollout_n, -1).contiguous()
        text_seq_len = captured_conditioning["text_seq_len"]
        text_hidden_states, text_attention_mask = self._pad_or_truncate_conditioning(
            text_hidden_states,
            text_attention_mask,
            conditioning_max_length,
        )

        # sigma_schedule is the same for all samples; store it once per sample for easy slicing.
        sigma_schedule_tensor = sigma_schedule.unsqueeze(0).expand(rollout_n, -1).contiguous()  # [B, steps+1]

        conditioning_tensors = {
            "all_latents": all_latents,
            "all_log_probs": all_log_probs,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "sigma_schedule": sigma_schedule_tensor,
            # Actual (unpadded) sequence lengths — 1D [B] tensors for DataProto concat.
            "text_seq_len": torch.tensor([text_seq_len] * rollout_n, dtype=torch.long),
        }
        if "negative_text_hidden_states" in captured_conditioning:
            negative_text_hidden_states = (
                captured_conditioning["negative_text_hidden_states"].expand(rollout_n, -1, -1).contiguous()
            )
            negative_text_attention_mask = (
                captured_conditioning["negative_text_attention_mask"].expand(rollout_n, -1).contiguous()
            )
            neg_seq_len = captured_conditioning["neg_seq_len"]
            negative_text_hidden_states, negative_text_attention_mask = self._pad_or_truncate_conditioning(
                negative_text_hidden_states,
                negative_text_attention_mask,
                conditioning_max_length,
            )
            conditioning_tensors["negative_text_hidden_states"] = negative_text_hidden_states
            conditioning_tensors["negative_text_attention_mask"] = negative_text_attention_mask
            conditioning_tensors["neg_seq_len"] = torch.tensor([neg_seq_len] * rollout_n, dtype=torch.long)
        if "image_hidden_states" in captured_conditioning:
            image_hidden_states = captured_conditioning["image_hidden_states"].expand(rollout_n, -1, -1).contiguous()
            image_attention_mask = captured_conditioning["image_attention_mask"].expand(rollout_n, -1).contiguous()
            img_seq_len = captured_conditioning["image_seq_len"]
            image_hidden_states, image_attention_mask = self._pad_or_truncate_conditioning(
                image_hidden_states,
                image_attention_mask,
                conditioning_max_length,
            )
            conditioning_tensors["image_hidden_states"] = image_hidden_states
            conditioning_tensors["image_attention_mask"] = image_attention_mask
            conditioning_tensors["image_seq_len"] = torch.tensor([img_seq_len] * rollout_n, dtype=torch.long)

        if _rank == 0:
            logger.info("Built-in Rollout Complete.")

        batch = DataProto.from_dict(
            tensors=conditioning_tensors,
            non_tensors={
                "all_images": all_images,
                "image_paths": np.array(image_paths, dtype=object),
            },
        )
        return batch

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = len(prompts)
        batch_prompts = prompts.chunk(chunks=batch_size // self.config.n)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    def release(self):
        pass

    def resume(self):
        pass

    def update_weights(self, data: DataProto) -> dict:
        pass
