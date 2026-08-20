#!/usr/bin/env bash
set -euo pipefail

TAR="$1"
EXT_ROOT="/data/kaichen/data/llava-onevision-2/180s_streaming_ext"
INDEX_DIR="/data/kaichen/data/llava-onevision-2/index"
LOG_DIR="/data/kaichen/data/llava-onevision-2/extract_logs"

part=$(basename "$(dirname "$TAR")")
shard=$(basename "$TAR" .tar)
marker="${INDEX_DIR}/${part}__${shard}.jsonl"
log="${LOG_DIR}/${part}__${shard}.log"

mkdir -p "$LOG_DIR" "$INDEX_DIR"

if [[ -f "$marker" ]]; then
  echo "[skip] $part/$shard already has index" >>"$LOG_DIR/skipped.log"
  exit 0
fi

python -u /tmp/extract_wds_shards.py \
  --tars "$TAR" \
  --ext_root "$EXT_ROOT" \
  --index_dir "$INDEX_DIR" \
  >"$log" 2>&1
