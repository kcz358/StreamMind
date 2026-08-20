#!/usr/bin/env bash
# Process the wds_180s parquet in fixed-size batches. For each batch:
#   1. cut mp4 clips into /local_nvme/wds/clips_stage (fast local ffmpeg)
#   2. gen_codec: cache-key = canonical blob path, IO = local mirror
#   3. azcopy sync /local_nvme/wds/clips_stage -> blob clips dir
#   4. rm local mp4 to free nvme for the next batch
#
# Env required:
#   AZCOPY_BASE_URL, AZCOPY_SAS_TOKEN
#
# Args:
#   BATCH_SIZE      rows per batch (default 50000)
#   START_ROW       first parquet row to process (default 0)
#   END_ROW         one past last row (default = parquet len; auto-detected)

: "${AZCOPY_BASE_URL:?}"
: "${AZCOPY_SAS_TOKEN:?}"

MANIFEST=/data/kaichen/data/MatchTime/manifests/wds_180s.parquet
SAMPLE_MAP=/data/kaichen/data/MatchTime/manifests/wds_180s.sample_map.jsonl
MATCHTIME_ROOT=/data/kaichen/data/MatchTime
BLOB_CLIP_DIR=/mnt/blob/kaichen/b200_node/data/MatchTime/clips
LOCAL_CLIP_DIR=/local_nvme/wds/clips_stage
CODEC_DIR=${MATCHTIME_ROOT}/codec_cache_p16
CODEC_MODULE=/data/kaichen/checkpoints/OV2/Stage5_SFT_4B_mimo_v15_lr2e-6_HF_ported
CUT_WORKERS=${CUT_WORKERS:-16}
CODEC_WORKERS=${CODEC_WORKERS:-16}

BATCH_SIZE=${BATCH_SIZE:-50000}
START_ROW=${START_ROW:-0}
if [[ -z "${END_ROW:-}" ]]; then
  END_ROW=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${MANIFEST}')))")
fi

LOG_DIR=/local_nvme/wds/batch_logs
DONE_FILE=${LOG_DIR}/batches_done.txt
mkdir -p "$LOG_DIR" "$LOCAL_CLIP_DIR"
touch "$DONE_FILE"

echo "[driver] parquet rows=${END_ROW}  batch=${BATCH_SIZE}  start=${START_ROW}"

s=$START_ROW
while (( s < END_ROW )); do
  e=$(( s + BATCH_SIZE ))
  if (( e > END_ROW )); then e=$END_ROW; fi
  tag=$(printf "b%08d_%08d" "$s" "$e")
  log="${LOG_DIR}/${tag}.log"

  if grep -qxF "$tag" "$DONE_FILE"; then
    echo "[driver] skip $tag (already done)"
    s=$e
    continue
  fi

  echo "[driver] batch ${tag}  $(date -Is)" | tee -a "$log"

  set +e
  echo "[cut]" | tee -a "$log"
  python3 -u /tmp/cut_wds_clips.py \
    --manifest "$MANIFEST" --sample_map "$SAMPLE_MAP" \
    --matchtime_root "$MATCHTIME_ROOT" \
    --clip_dir "$BLOB_CLIP_DIR" --local_clip_dir "$LOCAL_CLIP_DIR" \
    --workers "$CUT_WORKERS" --log_every 5000 \
    --start "$s" --limit "$BATCH_SIZE" >>"$log" 2>&1
  rc_cut=$?

  echo "[gen_codec] rc_cut=$rc_cut" | tee -a "$log"
  python3 -u /tmp/gen_codec.py \
    --manifest "$MANIFEST" --matchtime_root "$MATCHTIME_ROOT" \
    --clip_dir "$BLOB_CLIP_DIR" --local_clip_dir "$LOCAL_CLIP_DIR" \
    --codec_dir "$CODEC_DIR" --codec_module "$CODEC_MODULE" \
    --workers "$CODEC_WORKERS" --log_every 5000 \
    --start "$s" --limit "$BATCH_SIZE" >>"$log" 2>&1
  rc_codec=$?

  echo "[upload+purge] rc_codec=$rc_codec" | tee -a "$log"
  bash /tmp/upload_and_purge_clips.sh "$LOCAL_CLIP_DIR" b200_node/data/MatchTime/clips \
    "${LOG_DIR}/${tag}_azcopy.log"
  rc_up=$?
  set -e

  echo "[driver] batch ${tag} done rc_cut=$rc_cut rc_codec=$rc_codec rc_up=$rc_up $(date -Is)" | tee -a "$log"
  if [[ $rc_cut -eq 0 && $rc_codec -eq 0 && $rc_up -eq 0 ]]; then
    echo "$tag" >>"$DONE_FILE"
  fi
  s=$e
done

echo "[driver] all batches done $(date -Is)"
