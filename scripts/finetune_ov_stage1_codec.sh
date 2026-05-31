#!/bin/bash
# scripts/finetune_ov_stage1_codec.sh
set -e
cd "$(dirname "$0")/.."

export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ONLINE_CODEC_CACHE_DIR=/data/v-kaichen/azure_blob/data/MatchTime/codec_cache
export PATH="$PWD/.venv/bin:$PATH"

.venv/bin/torchrun --nproc_per_node 4 --master_port 16668 \
    ov_train/train.py \
    --stage 1 \
    --video_backend codec \
    --num_frames 16 \
    --split train \
    --max_samples 64 \
    --model_name_or_path /data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct \
    --output_dir output/ov_stage1_codec_smoke \
    --deepspeed scripts/zero3.json \
    --max_steps 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    --dataloader_num_workers 0 \
    --remove_unused_columns False
