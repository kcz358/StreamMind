#!/usr/bin/env bash
# Wrap run_batches.sh in an infinite respawn loop.
# Exits only when done_count >= TOTAL_BATCHES.
export AZCOPY_BASE_URL AZCOPY_SAS_TOKEN
BATCH_SIZE=${BATCH_SIZE:-50000}
START_ROW=${START_ROW:-0}
END_ROW=${END_ROW:-1000000}
DRIVER=/tmp/run_batches.sh
LOG_ROOT=/local_nvme/wds
DONE_FILE=${LOG_ROOT}/batch_logs/batches_done.txt
LOOP_LOG=${LOG_ROOT}/respawn.log
TOTAL_BATCHES=$(( (END_ROW - START_ROW + BATCH_SIZE - 1) / BATCH_SIZE ))

mkdir -p "$(dirname "$DONE_FILE")"
touch "$DONE_FILE"

echo "[respawn] start $(date -Is)  total=${TOTAL_BATCHES}" >>"$LOOP_LOG"

while true; do
  done=$(wc -l <"$DONE_FILE" 2>/dev/null || echo 0)
  if (( done >= TOTAL_BATCHES )); then
    echo "[respawn] all done ($done/$TOTAL_BATCHES) $(date -Is)" >>"$LOOP_LOG"
    break
  fi
  ts=$(date +%Y%m%d_%H%M%S)
  echo "[respawn] launching driver at $ts  done=$done/$TOTAL_BATCHES" >>"$LOOP_LOG"
  BATCH_SIZE="$BATCH_SIZE" START_ROW="$START_ROW" END_ROW="$END_ROW" \
    bash "$DRIVER" >"${LOG_ROOT}/batch_driver_${ts}.log" 2>&1
  rc=$?
  echo "[respawn] driver exited rc=$rc at $(date -Is)" >>"$LOOP_LOG"
  sleep 30
done
