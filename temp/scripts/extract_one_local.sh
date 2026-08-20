#!/usr/bin/env bash
set -euo pipefail

TAR="$1"
EXT_ROOT="/local_nvme/wds/180s_streaming_ext"
INDEX_DIR="/local_nvme/wds/index"
LOG_DIR="/local_nvme/wds/extract_logs"

part=$(basename "$(dirname "$TAR")")
shard=$(basename "$TAR" .tar)
marker="${INDEX_DIR}/${part}__${shard}.jsonl"
log="${LOG_DIR}/${part}__${shard}.log"

mkdir -p "$LOG_DIR" "$INDEX_DIR"

if [[ -f "$marker" ]]; then
  echo "[skip] $part/$shard already extracted" >>"$LOG_DIR/skipped.log"
  [[ -f "$TAR" ]] && rm -f "$TAR"
  exit 0
fi

python -u /tmp/extract_wds_shards.py \
  --tars "$TAR" \
  --ext_root "$EXT_ROOT" \
  --index_dir "$INDEX_DIR" \
  >"$log" 2>&1

if [[ -f "$marker" ]]; then
  rm -f "$TAR"
  echo "[rm] $TAR" >>"$LOG_DIR/removed.log"
fi
