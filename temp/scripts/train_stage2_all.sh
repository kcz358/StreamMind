#!/usr/bin/env bash
# Stage-2 CLS training on merged train_all.parquet, init_from stage1.
set -euo pipefail

: "${STAGE1_CKPT:?STAGE1_CKPT must point to the final stage1 checkpoint dir}"

REPO=/workspace/StreamMind
export MATCHTIME_ROOT=/data/kaichen/data/MatchTime
export SOCCER_MANIFEST_DIR=${MATCHTIME_ROOT}/manifests
export STREAMMIND_CLIP_CACHE=/mnt/blob/kaichen/b200_node/data/MatchTime/clips
export ONLINE_CODEC_CACHE_DIR=${MATCHTIME_ROOT}/codec_cache_p16
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=lo

MODEL_PATH=/data/kaichen/checkpoints/OV2/Stage5_SFT_4B_mimo_v15_lr2e-6_HF_ported
RUN_TAG="p16_4b_stage2_train_all_$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR=/data/kaichen/StreamMind/${RUN_TAG}

: "${WANDB_API_KEY:?}"
export WANDB_PROJECT=streammind-paper
export WANDB_NAME=${RUN_TAG}

mkdir -p "${OUTPUT_DIR}"
LOGFILE="${OUTPUT_DIR}/train.log"

cd "${REPO}"

torchrun \
  --nproc_per_node 8 --nnodes 1 --master_addr 127.0.0.1 --master_port 26200 \
  --redirects 3 --tee 3 \
  --log-dir ${OUTPUT_DIR}/torchrun_logs \
  ov_train/train.py \
    --soccer_dataset True \
    --soccer_dataset_train_cls True \
    --init_from_ckpt "${STAGE1_CKPT}" \
    --video_backend codec \
    --num_frames 16 \
    --data_type train_all \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --deepspeed scripts/zero2.json \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 5 \
    --report_to wandb \
    --run_name "${WANDB_NAME}" \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 4 \
    --dataloader_persistent_workers True \
    --ddp_timeout 7200 \
    --remove_unused_columns False \
  2>&1 | tee "${LOGFILE}"
