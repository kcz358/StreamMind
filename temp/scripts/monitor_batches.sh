#!/usr/bin/env bash
# Watchdog: keep the batch driver alive until all 20 batches are done.
# Every SLEEP seconds:
#   * count batches in ${DONE_FILE}; exit if >= TOTAL_BATCHES
#   * if no run_batches.sh process exists, relaunch it
set -u

BATCH_SIZE=${BATCH_SIZE:-50000}
START_ROW=${START_ROW:-0}
END_ROW=${END_ROW:-1000000}
DRIVER=/tmp/run_batches.sh
LOG_ROOT=/local_nvme/wds
DONE_FILE=${LOG_ROOT}/batch_logs/batches_done.txt
MON_LOG=${LOG_ROOT}/monitor.log
SLEEP=${SLEEP:-300}
TOTAL_BATCHES=$(( (END_ROW - START_ROW + BATCH_SIZE - 1) / BATCH_SIZE ))

mkdir -p "$(dirname "$DONE_FILE")"
touch "$DONE_FILE"

echo "[monitor] start $(date -Is)  total=${TOTAL_BATCHES}  sleep=${SLEEP}s" >>"$MON_LOG"

while true; do
  done=$(wc -l <"$DONE_FILE" 2>/dev/null || echo 0)
  echo "[monitor] $(date -Is)  done=${done}/${TOTAL_BATCHES}" >>"$MON_LOG"
  if (( done >= TOTAL_BATCHES )); then
    echo "[monitor] all batches done, exiting" >>"$MON_LOG"
    exit 0
  fi
  if ! pgrep -f run_batches.sh >/dev/null 2>&1; then
    ts=$(date +%Y%m%d_%H%M%S)
    echo "[monitor] driver missing, relaunching $ts" >>"$MON_LOG"
    setsid nohup env AZCOPY_BASE_URL="$AZCOPY_BASE_URL" AZCOPY_SAS_TOKEN="$AZCOPY_SAS_TOKEN" \
      BATCH_SIZE="$BATCH_SIZE" START_ROW="$START_ROW" END_ROW="$END_ROW" \
      bash "$DRIVER" \
      >"${LOG_ROOT}/batch_driver_${ts}.log" 2>&1 </dev/null &
    disown || true
    sleep 5
  fi
  sleep "$SLEEP"
done
