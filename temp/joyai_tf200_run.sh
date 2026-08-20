#!/bin/bash
# Fallback: teacher-forced JoyAI trigger evaluation with a 200 s window.
cd /root
OUT=/data/kaichen/logs/streaming_trigger_eval_20260722/joyai_tf200
mkdir -p $OUT
setsid torchrun --nproc_per_node=8 --master_port=29547 eval_joyai_trigger.py \
  --manifest /data/kaichen/data/MatchTime/manifests/valid.parquet \
  --chunk_seconds 200 --batch_size 1 --tolerance_seconds 1 \
  --progress_dir $OUT >> $OUT/run.log 2>&1 < /dev/null &
