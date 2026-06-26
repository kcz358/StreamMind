#!/usr/bin/env bash
# Container-side entrypoint for OneVision×StreamMind two-stage training on Volcano.
# Run on every node. Volcano pytorch plugin sets RANK/MASTER_ADDR/MASTER_PORT/NODES.
#
# Required env (set by job_template.yaml):
#   STAGE              1 or 2
#   VIDEO_BACKEND      frames | codec
#   MODEL_PATH         HF model dir (for p16 codec, use a 4B *_ported ckpt)
#   MATCHTIME_ROOT     PVC dir containing dataset/ + features_video/
#   OUTPUT_DIR         where checkpoints + logs go
#   NUM_FRAMES         e.g. 16
#   MAX_STEPS          e.g. 100 (smoke) or -1 for full epoch
#   LR                 e.g. 2e-5 (stage1) / 1e-4 (stage2)
#   SPLIT              train | valid
#   DEEPSPEED_CONFIG   path to zero2.json / zero3.json
#   GRAD_ACCUM         gradient_accumulation_steps
#   SAVE_STEPS         save checkpoint every N steps
#   LOGGING_STEPS      log loss every N steps
#
# Container env (provided by Volcano pytorch plugin):
#   RANK, MASTER_ADDR, MASTER_PORT, NODES, GPUS_PER_NODE
set -euo pipefail

: "${STAGE:?}"
: "${VIDEO_BACKEND:?}"
: "${MODEL_PATH:?}"
: "${MATCHTIME_ROOT:?}"
: "${OUTPUT_DIR:?}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NODES:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-26000}"

mkdir -p "${OUTPUT_DIR}"
TM="$(date '+%Y-%m-%d_%H:%M:%S')"
LOGFILE="${OUTPUT_DIR}/run_${TM}_stage${STAGE}_${VIDEO_BACKEND}.log"

# Codec backend uses precomputed cache on PVC. Keep clips on blob if desired;
# p16 training only needs the clip path string to compute the cache key.
if [[ "${VIDEO_BACKEND}" == "codec" ]]; then
  export ONLINE_CODEC_CACHE_DIR="${ONLINE_CODEC_CACHE_DIR:-${MATCHTIME_ROOT}/codec_cache_p16}"
  mkdir -p "${ONLINE_CODEC_CACHE_DIR}"
fi

export STREAMMIND_CLIP_CACHE="${STREAMMIND_CLIP_CACHE:-/mnt/blob/kaichen/b200_node/data/MatchTime/clips}"
export SOCCER_MANIFEST_DIR="${SOCCER_MANIFEST_DIR:-${MATCHTIME_ROOT}/manifests}"

# StreamOneVision wrapper reads pad token from processor; we mark it offline
# only if the model dir is fully self-contained (it is for HF snapshots).
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"

# Stage 2 cannot use gradient checkpointing when base is frozen under ZeRO-3
# (ZeRO-3 releases base params after forward, GC recompute sees empty shards).
# For ZeRO-2 + frozen base, GC works fine. Default to True; entry can override.
if [[ "${STAGE}" == "2" && "${DEEPSPEED_CONFIG}" == *"zero3"* ]]; then
  GC_FLAG=False
else
  GC_FLAG=True
fi

MAX_STEPS_FLAG=()
if [[ "${MAX_STEPS:--1}" != "-1" ]]; then
  MAX_STEPS_FLAG=(--max_steps "${MAX_STEPS}")
fi

echo "=================================================="
echo "STAGE          = ${STAGE}"
echo "VIDEO_BACKEND  = ${VIDEO_BACKEND}"
echo "MODEL_PATH     = ${MODEL_PATH}"
echo "MATCHTIME_ROOT = ${MATCHTIME_ROOT}"
echo "CLIP_CACHE     = ${STREAMMIND_CLIP_CACHE}"
echo "CODEC_CACHE    = ${ONLINE_CODEC_CACHE_DIR:-}"
echo "MANIFEST_DIR   = ${SOCCER_MANIFEST_DIR}"
echo "OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "DEEPSPEED_CFG  = ${DEEPSPEED_CONFIG}"
echo "rank ${NODE_RANK}/${NNODES}  master=${MASTER_ADDR}:${MASTER_PORT}  gpus=${GPUS_PER_NODE}"
echo "=================================================="

if [[ -n "${WANDB_API_KEY:-}" && "${WANDB_API_KEY}" != "none" ]]; then
  REPORT_TO=wandb
else
  REPORT_TO=none
fi

RESUME_FLAG=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  RESUME_FLAG=(--resume_from_checkpoint "${RESUME_FROM}")
fi

TRAIN_MODE_FLAG=()
if [[ "${STAGE}" == "1" ]]; then
  TRAIN_MODE_FLAG=(--soccer_dataset_train_llm True)
elif [[ "${STAGE}" == "2" ]]; then
  TRAIN_MODE_FLAG=(--soccer_dataset_train_cls True)
else
  echo "ERROR: STAGE must be 1 or 2, got ${STAGE}" >&2
  exit 2
fi

torchrun \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  ov_train/train.py \
    --soccer_dataset True \
    "${TRAIN_MODE_FLAG[@]}" \
    "${RESUME_FLAG[@]}" \
    --video_backend "${VIDEO_BACKEND}" \
    --num_frames "${NUM_FRAMES:-16}" \
    --data_type "${SPLIT:-train}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    "${MAX_STEPS_FLAG[@]}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
    --learning_rate "${LR}" \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing ${GC_FLAG} \
    --logging_steps "${LOGGING_STEPS:-1}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS:-500}" \
    --save_total_limit 5 \
    --report_to "${REPORT_TO}" \
    --run_name "${WANDB_NAME:-}" \
    --dataloader_num_workers 2 \
    --dataloader_prefetch_factor 4 \
    --dataloader_persistent_workers True \
    --ddp_timeout 7200 \
    --remove_unused_columns False \
  2>&1 | tee "${LOGFILE}"
