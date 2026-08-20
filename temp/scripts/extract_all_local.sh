#!/usr/bin/env bash
set -euo pipefail

ROOT="/local_nvme/wds/180s_streaming/180s_streaming"
LOG_DIR="/local_nvme/wds/extract_logs"
JOBS="${JOBS:-4}"

mkdir -p "$LOG_DIR"

find "$ROOT" -maxdepth 2 -name "*.tar" | sort >"$LOG_DIR/all_tars.txt"
n=$(wc -l <"$LOG_DIR/all_tars.txt")
echo "[launcher] queued $n tars, jobs=$JOBS, $(date -Is)" >>"$LOG_DIR/launcher.log"

xargs -a "$LOG_DIR/all_tars.txt" -n1 -P"$JOBS" bash /tmp/extract_one_local.sh \
  >>"$LOG_DIR/launcher.log" 2>&1

echo "[launcher] finished $(date -Is)" >>"$LOG_DIR/launcher.log"
