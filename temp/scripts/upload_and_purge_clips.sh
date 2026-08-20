#!/usr/bin/env bash
# Sync a local clip staging dir into the canonical blob clip dir and delete
# the local originals on success. Both dirs mirror the same relpath layout.
#
# Env:
#   AZCOPY_BASE_URL   e.g. https://mcgvisionflowsa.blob.core.windows.net/kaichen
#   AZCOPY_SAS_TOKEN  container SAS starting with '?'
set -euo pipefail

LOCAL_DIR="${1:?usage: $0 <local_clip_dir> <blob_relpath>}"
BLOB_RELPATH="${2:?}"
LOG="${3:-/local_nvme/wds/azcopy_clip_upload.log}"

: "${AZCOPY_BASE_URL:?}"
: "${AZCOPY_SAS_TOKEN:?}"

export AZCOPY_CONCURRENCY_VALUE=64
export AZCOPY_BUFFER_GB=8
export AZCOPY_LOG_LOCATION=/local_nvme/wds/azcopy_logs_clip
export AZCOPY_JOB_PLAN_LOCATION=/local_nvme/wds/azcopy_plans_clip
mkdir -p "$AZCOPY_LOG_LOCATION" "$AZCOPY_JOB_PLAN_LOCATION"

DST="${AZCOPY_BASE_URL}/${BLOB_RELPATH}/${AZCOPY_SAS_TOKEN}"

azcopy copy "${LOCAL_DIR%/}/*" "$DST" \
  --recursive \
  --overwrite=ifSourceNewer \
  --log-level=INFO \
  >>"$LOG" 2>&1

find "$LOCAL_DIR" -type f -name '*.mp4' -delete
