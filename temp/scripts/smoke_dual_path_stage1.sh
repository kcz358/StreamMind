#!/usr/bin/env bash
set -euo pipefail

cd /root/StreamMind
export MATCHTIME_ROOT=/data/kaichen/data/MatchTime
export SOCCER_MANIFEST_DIR=${MATCHTIME_ROOT}/manifests
export STREAMMIND_CLIP_CACHE=/mnt/blob/kaichen/b200_node/data/MatchTime/clips
export ONLINE_CODEC_CACHE_DIR=${MATCHTIME_ROOT}/codec_cache_p16
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=lo

torchrun \
  --nproc_per_node 8 --nnodes 1 --master_addr 127.0.0.1 --master_port 26300 \
  ov_train/train.py \
    --soccer_dataset True \
    --soccer_dataset_train_llm True \
    --video_backend codec \
    --num_frames 16 \
    --data_type train_all \
    --model_name_or_path /data/kaichen/checkpoints/OV2/Stage5_SFT_4B_mimo_v15_lr2e-6_HF_ported \
    --output_dir /tmp/dual_path_stage1_smoke \
    --deepspeed scripts/zero2.json \
    --max_steps 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    --dataloader_num_workers 2 \
    --dataloader_prefetch_factor 2 \
    --dataloader_persistent_workers True \
    --ddp_timeout 7200 \
    --remove_unused_columns False
