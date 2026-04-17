<p align="center">
<h1 align="center"> Recipe: Reinforcement learning for generative models using FSDP as the backend (DanceGRPO) </h1>

## 1. Environment installation ##

\[You are advised to use the matching environment version during model development.\]

```shell
# Importing CAN Environment Variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# Creating the python3.11 conda environment
conda create -n test python=3.11
conda activate test

# Download and Install Verl
git clone https://github.com/volcengine/verl.git
cd verl
git checkout v0.7.0
pip install -r requirements-npu.txt
pip install -r requirements.txt
pip install -v -e .
cd ..

# Update the recipe directory.
git clone https://github.com/verl-project/verl-recipe.git
mkdir verl/recipe/dance_grpo
cp -rf verl-recipe/dance_grpo/dance_grpo_ fsdp verl/recipe/dance_grpo/

# Installing the HPSv3 Scoring Model
git clone https://github.com/MizzenAI/HPSv3.git
cd HPSv3
git checkout upgrade_transformers_version
pip install -e .
cd ..

# Installing Other Packages
pip install diffusers==0.35.1 peft==0.17.1 torch_npu==2.7.1 loguru==0.7.3 opencv-python-headless==4.10.0.84 tf-keras matplotlib==3.8.4

cd verl

# The directory structure after the preparation is as follows:
# HPSv3
# verl-recipe
# verl
# ├── recipe
#     ├── dance_grpo
#         ├── dance_grpo_fsdp
```

## 2. Dataset preparation ##

Reference ` verl/recipe/dance_grpo/dance_grpo_fsdp/data/prompt.json ` In the example provided in, you can replace the customized prompt text and run the following command to generate a parquet file:

## 3. Training model preparation ##

Mammothmoda2 model download address:

https://huggingface.co/bytedance-research/MammothModa2-Dev

## 4. Scoring Model preparation ##

1. Download the HPSv3 model: https://huggingface.co/MizzenAI/HPSv3/tree/main
In the training script, change the value of +actor_rollout_ref.model.reward_model_path to the HPSv3 weight path.
```shell
+actor_rollout_ref.model.reward_model_path=/home/CKPT/HPSv3/HPSv3.safetensors
```

2. Download the Qwen2.5VL 7B model: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main
Modify the HPSv3/hpsv3/config/HPSv3_7B.yaml file to configure the directory for the Qwen2-VL-7B model.
```shell
model_name_or_path: "/home/CKPT/Qwen2-VL-7B-Instruct"
```


## 5. Model RL training ##

Modifying Parameters in the verl/recipe/dance_grpo/dance_grpo_fsdp/run_verl_dance.sh RL Training Script

### 1. Environment Variables

| Parameter                     | Value                      | Description                                                                                       |
| ----------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------- |
| `ASCEND_RT_VISIBLE_DEVICES`   | `12,13,14,15`              | Specifies visible NPU device IDs for Ascend accelerators                                          |
| `PYTORCH_NPU_ALLOC_CONF`      | `expandable_segments:True` | PyTorch NPU memory allocation config, enables expandable segments for optimized memory management |
| `HYDRA_FULL_ERROR`            | `1`                        | Enables full error message output from Hydra configuration framework                              |
| `HCCL_HOST_SOCKET_PORT_RANGE` | `auto`                     | HCCL (Huawei Collective Communication Library) host socket port range auto-allocation             |
| `HCCL_NPU_SOCKET_PORT_RANGE`  | `auto`                     | HCCL NPU socket port range auto-allocation                                                        |
| `DISABLE_L2_CACHE`            | `1`                        | Disables L2 cache                                                                                 |

### 2. Data Configuration

| Parameter                      | Value         | Description                                                         |
| ------------------------------ | ------------- | ------------------------------------------------------------------- |
| `data.train_files`             | `$train_data` | Path to training data file                                          |
| `data.val_files`               | `$test_data`  | Path to validation data file                                        |
| `data.train_batch_size`        | `1`           | Training batch size (samples per GPU)                               |
| `data.max_prompt_length`       | `4096`        | Maximum token length for input prompts                              |
| `data.max_response_length`     | `128`         | Maximum token length for model-generated responses                  |
| `data.filter_overlong_prompts` | `True`        | Whether to filter prompts exceeding max length                      |
| `data.truncation`              | `error`       | Truncation strategy; `error` raises exception instead of truncating |

### 3. Model Configuration

| Parameter                                               | Value         | Description                                        |
| ------------------------------------------------------- | ------------- | -------------------------------------------------- |
| `actor_rollout_ref.model.path`                          | `$model_path` | Path to pretrained model weights                   |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `False`       | Enable gradient checkpointing to save memory       |
| `actor_rollout_ref.model.enable_activation_offload`     | `False`       | Enable offloading activations to CPU               |
| `actor_rollout_ref.model.use_remove_padding`            | `False`       | Remove padding tokens for computation optimization |

### 4. Optimizer Configuration

| Parameter                                                       | Value    | Description                                     |
| --------------------------------------------------------------- | -------- | ----------------------------------------------- |
| `actor_rollout_ref.actor.optim.lr`                              | `5e-8`   | Learning rate                                   |
| `actor_rollout_ref.actor.optim.weight_decay`                    | `0.01`   | Weight decay coefficient for L2 regularization  |
| `actor_rollout_ref.actor.optim.lr_scheduler.name`               | `cosine` | Learning rate scheduler type (cosine annealing) |
| `actor_rollout_ref.actor.optim.lr_scheduler.num_warmup_steps`   | `1000`   | Warmup steps for learning rate scheduler        |
| `actor_rollout_ref.actor.optim.lr_scheduler.num_training_steps` | `10000`  | Total training steps                            |
| `actor_rollout_ref.actor.optim.lr_scheduler.num_cycles`         | `1`      | Number of cycles for cosine scheduler           |

### 5. PPO/GRPO Algorithm Configuration

| Parameter                                   | Value   | Description                                                   |
| ------------------------------------------- | ------- | ------------------------------------------------------------- |
| `algorithm.adv_estimator`                   | `grpo`  | Advantage estimator type (Group Relative Policy Optimization) |
| `algorithm.use_kl_in_reward`                | `False` | Whether to add KL divergence penalty to reward                |
| `actor_rollout_ref.actor.ppo_adv_clip_max`  | `10.0`  | Maximum clip value for PPO advantage function                 |
| `actor_rollout_ref.actor.ppo_kl_coeff`      | `1.0`   | KL divergence penalty coefficient                             |
| `actor_rollout_ref.actor.ppo_max_grad_norm` | `1.0`   | Maximum gradient norm for gradient clipping                   |
| `actor_rollout_ref.actor.clip_range`        | `1e-4`  | PPO policy clipping range                                     |
| `actor_rollout_ref.actor.shift`             | `1.0`   | Timestep shift for Diffusion models                           |
| `actor_rollout_ref.actor.timestep_fraction` | `1`     | Timestep sampling fraction                                    |
| `actor_rollout_ref.actor.sampling_steps`    | `10`    | Diffusion sampling steps                                      |

### 6. Batch Size Configuration

| Parameter                                             | Value | Description                                      |
| ----------------------------------------------------- | ----- | ------------------------------------------------ |
| `actor_rollout_ref.actor.micro_batch_size`            | `2`   | Micro batch size for actor training              |
| `actor_rollout_ref.actor.ppo_mini_batch_size`         | `1`   | PPO mini batch size                              |
| `actor_rollout_ref.actor.ppo_micro_batch_size`        | `1`   | PPO micro batch size                             |
| `actor_rollout_ref.rollout.micro_batch_size`          | `2`   | Micro batch size for rollout inference           |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size` | `1`   | Micro batch size for log probability computation |

### 7. FSDP Distributed Configuration

| Parameter                                               | Value   | Description                                                |
| ------------------------------------------------------- | ------- | ---------------------------------------------------------- |
| `actor_rollout_ref.actor.strategy`                      | `fsdp`  | Training strategy using FSDP (Fully Sharded Data Parallel) |
| `actor_rollout_ref.actor.fsdp_config.param_offload`     | `False` | Offload model parameters to CPU                            |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | `False` | Offload optimizer states to CPU                            |
| `actor_rollout_ref.actor.use_torch_compile`             | `True`  | Use torch.compile for acceleration                         |

### 8. Rollout Configuration

| Parameter                                              | Value  | Description                                    |
| ------------------------------------------------------ | ------ | ---------------------------------------------- |
| `actor_rollout_ref.rollout.mode`                       | `sync` | Rollout mode; `sync` for synchronous execution |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `1`    | Tensor model parallelism size                  |
| `actor_rollout_ref.rollout.gpu_memory_utilization`     | `0.30` | GPU memory utilization limit (30%)             |
| `actor_rollout_ref.rollout.n`                          | `1`    | Number of responses generated per prompt       |
| `actor_rollout_ref.rollout.latent_w`                   | `128`  | Diffusion latent space width                   |
| `actor_rollout_ref.rollout.latent_h`                   | `128`  | Diffusion latent space height                  |
| `actor_rollout_ref.rollout.init_same_noise`            | `True` | Use same initial noise for generation          |

### 9. Trainer Configuration

| Parameter                      | Value                     | Description                                   |
| ------------------------------ | ------------------------- | --------------------------------------------- |
| `trainer.critic_warmup`        | `0`                       | Critic warmup steps                           |
| `trainer.logger`               | `console`                 | Logger type (console output)                  |
| `trainer.val_before_train`     | `False`                   | Run validation before training                |
| `trainer.project_name`         | `mammothmoda2_project`    | Project name for logging                      |
| `trainer.experiment_name`      | `mammothmoda2_experiment` | Experiment name                               |
| `trainer.n_gpus_per_node`      | `1`                       | Number of GPUs per node                       |
| `trainer.nnodes`               | `1`                       | Number of nodes                               |
| `trainer.save_freq`            | `-1`                      | Model save frequency; `-1` disables auto-save |
| `trainer.test_freq`            | `1`                       | Test/validation frequency (every N epochs)    |
| `trainer.total_epochs`         | `2`                       | Total training epochs                         |
| `trainer.total_training_steps` | `2`                       | Total training steps                          |
| `trainer.device`               | `npu`                     | Training device type (NPU)                    |

### 10. Hydra Configuration

| Parameter       | Value               | Description                   |
| --------------- | ------------------- | ----------------------------- |
| `--config-path` | `config`            | Hydra configuration directory |
| `--config-name` | `dance_ppo_trainer` | Hydra configuration filename  |

Start RL training.

```shell
# cd verl

bash recipe/dance_grpo/dance_grpo_fsdp/run_verl_dance.sh
```