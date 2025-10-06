#!/usr/bin/env bash


export RAY_ADDRESS="http://10.42.150.208:8265"
WORKING_DIR=/path/to/verl/
RUNTIME_ENV=runtime_env.yaml

MODEL_PATH=/itdd-pfs/LLM/public-models/Qwen2.5-3B-Base

TRAIN_FILE1=/path/to/data/_dapo.parquet
TRAIN_FILE2=/path/to/data/_MATH.parquet

TEST_FILE1=/path/to/data/_aime_2024.parquet
TEST_FILE2=/path/to/data/_aime_2025.parquet
TEST_FILE3=/path/to/data/_amc2023.parquet
TEST_FILE3=/path/to/data/_math500.parquet



ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python -u -m recipe.gspo_ib.main_ib \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode='gspo' \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=14240 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.shuffle=True \
    data.train_files=[${TRAIN_FILE1},${TRAIN_FILE2}] \
    data.val_files=[${TEST_FILE1},${TEST_FILE2},${TEST_FILE3}] \
    data.train_batch_size=512 \
    data.shuffle=True \
    trainer.test_freq=20 \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    trainer.resume_mode=auto \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='RL-GSPO-IB' \
    trainer.experiment_name='gspo-lonspo-Qwen2.5-3B-Base-RL' \
    trainer.n_gpus_per_node=8 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=10000 \
    trainer.total_epochs=10 \
    reward_model.reward_manager=gspoib \
    custom_reward_function.path=/path/to/reward_function.py \
    custom_reward_function.name=compute_score \
