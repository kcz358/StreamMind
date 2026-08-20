#!/usr/bin/env bash
# Upload the extracted wds frame tree (/local_nvme/wds/180s_streaming_ext/*)
# into the shared blob at b200_node/data/ov2_streaming/frames_ext/. Each
# top-level part directory is uploaded in its own azcopy job so we can
# resume on failure per-part without redoing already-completed uploads.
#
# Env: AZCOPY_BASE_URL, AZCOPY_SAS_TOKEN
set -u

: "${AZCOPY_BASE_URL:?}"
: "${AZCOPY_SAS_TOKEN:?}"

SRC_ROOT=/local_nvme/wds/180s_streaming_ext
DST_RELPATH=b200_node/data/ov2_streaming/frames_ext
LOG_DIR=/local_nvme/wds/azcopy_frames_ext_logs
PLAN_DIR=/local_nvme/wds/azcopy_frames_ext_plans

mkdir -p "$LOG_DIR" "$PLAN_DIR"
export AZCOPY_LOG_LOCATION="$LOG_DIR"
export AZCOPY_JOB_PLAN_LOCATION="$PLAN_DIR"

echo "[frames_ext] launcher start $(date -Is)" >>"$LOG_DIR/launcher.log"

for part in "$SRC_ROOT"/*/; do
  name=$(basename "$part")
  DST="${AZCOPY_BASE_URL}/${DST_RELPATH}/${name}/${AZCOPY_SAS_TOKEN}"
  echo "[frames_ext] $(date -Is)  uploading ${name}" >>"$LOG_DIR/launcher.log"
  azcopy sync "${part%/}" "$DST" \
    --recursive \
    --delete-destination=false \
    --log-level=INFO \
    >>"$LOG_DIR/${name}.log" 2>&1
  rc=$?
  echo "[frames_ext] $(date -Is)  ${name} rc=$rc" >>"$LOG_DIR/launcher.log"
done

echo "[frames_ext] launcher done $(date -Is)" >>"$LOG_DIR/launcher.log"
