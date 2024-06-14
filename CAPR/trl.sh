#!/bin/bash

# SF_MODEL='/data4/share_nlp/data/luannd/pretrained_model/bloom-560m'
# RW_MODEL='/data4/share_nlp/data/luannd/pretrained_model/bloom-1b1'

accelerate launch --multi_gpu --num_machines 1 --num_processes 4 TRL_training.py
    # --log_with=tensorboard \
    # --model_name=$SF_MODEL \
    # --reward_model_name=$RW_MODEL \
    # --adafactor=False \
    # --tokenizer_name=$SF_MODEL \
    # --save_freq=100 --output_max_length=128 \
    # --batch_size=1 --gradient_accumulation_steps=8 \
    # --batched_gen=True --ppo_epochs=4 \
    # --seed=0 --learning_rate=1.4e-5 \
    # --early_stopping=True \
    # --output_dir=./test_rl_bloom
