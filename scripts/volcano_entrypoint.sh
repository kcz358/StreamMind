#!/usr/bin/env bash
# Container-side entrypoint for OneVision×StreamMind two-stage training on Volcano.
# Run on every node. Volcano pytorch plugin sets RANK/MASTER_ADDR/MASTER_PORT/NODES.
#
# Required env (set by job_template.yaml):
#   STAGE              1 or 2
#   VIDEO_BACKEND      frames | codec
#   MODEL_PATH         HF model dir (LLaVA-OneVision-2-8B-Instruct)
#   MATCHTIME_ROOT     PVC dir containing dataset/ + features_video/
#   OUTPUT_DIR         where checkpoints + logs go
#   NUM_FRAMES         e.g. 16
#   MAX_STEPS          e.g. 100 (smoke) or -1 for full epoch
#   MAX_SAMPLES        empty for full dataset, or int to cap
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

# Codec backend needs cv-preinfer on PATH; image already has it via pip.
# Cache dirs live on PVC so they persist + are shared across runs.
if [[ "${VIDEO_BACKEND}" == "codec" ]]; then
  export ONLINE_CODEC_CACHE_DIR="${MATCHTIME_ROOT}/codec_cache"
  mkdir -p "${ONLINE_CODEC_CACHE_DIR}"
fi

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

# Optional caps (empty -> drop the flag).
MAX_SAMPLES_FLAG=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_FLAG=(--max_samples "${MAX_SAMPLES}")
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
echo "OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "DEEPSPEED_CFG  = ${DEEPSPEED_CONFIG}"
echo "rank ${NODE_RANK}/${NNODES}  master=${MASTER_ADDR}:${MASTER_PORT}  gpus=${GPUS_PER_NODE}"
echo "=================================================="

if [[ -n "${WANDB_API_KEY:-}" && "${WANDB_API_KEY}" != "none" ]]; then
  REPORT_TO=wandb
else
  REPORT_TO=none
fi

torchrun \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  ov_train/train.py \
    --stage "${STAGE}" \
    --video_backend "${VIDEO_BACKEND}" \
    --num_frames "${NUM_FRAMES:-16}" \
    --split "${SPLIT:-train}" \
    "${MAX_SAMPLES_FLAG[@]}" \
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
    --remove_unused_columns False \
  2>&1 | tee "${LOGFILE}"
