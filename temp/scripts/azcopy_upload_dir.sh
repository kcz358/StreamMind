#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$1"           # /data/kaichen/data/MatchTime/core
DEST_NAME="$2"         # core
LOG="$3"

BASE="${AZCOPY_BASE_URL:?}"
SAS="${AZCOPY_SAS_TOKEN:?}"

export AZCOPY_CONCURRENCY_VALUE=64
export AZCOPY_BUFFER_GB=8
export AZCOPY_LOG_LOCATION="/local_nvme/wds/azcopy_logs_upload_${DEST_NAME}"
export AZCOPY_JOB_PLAN_LOCATION="/local_nvme/wds/azcopy_plans_upload_${DEST_NAME}"
mkdir -p "$AZCOPY_LOG_LOCATION" "$AZCOPY_JOB_PLAN_LOCATION"

DST="${BASE}/b200_node/data/MatchTime/${DEST_NAME}/${SAS}"

azcopy copy "${SRC_DIR%/}/*" "$DST" \
  --recursive \
  --overwrite=ifSourceNewer \
  --log-level=INFO \
  >"$LOG" 2>&1
