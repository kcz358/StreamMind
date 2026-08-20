#!/usr/bin/env bash
# Smoke train on the wds_180s parquet using 8 GPUs, ZeRO-2, codec backend.
# No wandb, no ckpt saving. Runs a few steps to confirm the pipeline works
# end-to-end (dataset build -> codec cache hit -> forward/backward).
set -euo pipefail

REPO=/workspace/StreamMind
export MATCHTIME_ROOT=/data/kaichen/data/MatchTime
export SOCCER_MANIFEST_DIR=${MATCHTIME_ROOT}/manifests
export STREAMMIND_CLIP_CACHE=/mnt/blob/kaichen/b200_node/data/MatchTime/clips
export ONLINE_CODEC_CACHE_DIR=${MATCHTIME_ROOT}/codec_cache_p16
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"

MODEL_PATH=/data/kaichen/checkpoints/OV2/Stage5_SFT_4B_mimo_v15_lr2e-6_HF_ported
OUTPUT_DIR=/tmp/smoke_wds_train

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

cd "${REPO}"

torchrun \
  --nproc_per_node 8 --nnodes 1 --master_addr 127.0.0.1 --master_port 26010 \
  ov_train/train.py \
    --soccer_dataset True \
    --soccer_dataset_train_llm True \
    --video_backend codec \
    --num_frames 16 \
    --data_type wds_180s \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --deepspeed scripts/zero2.json \
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
    --dataloader_num_workers 2 \
    --dataloader_prefetch_factor 4 \
    --dataloader_persistent_workers True \
    --ddp_timeout 7200 \
    --remove_unused_columns False

rm -rf "${OUTPUT_DIR}"
