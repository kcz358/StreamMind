#!/bin/bash
# Run 1: official JoyAI agent loop, T_s=100 window, no summary memory.
cd /root/joyai_agent
OUT=/data/kaichen/logs/streaming_trigger_eval_20260722/joyai_agent_ts100
mkdir -p $OUT
for rank in 0 1 2 3 4 5 6 7; do
  setsid .venv/bin/python eval_joyai_agent.py \
    --manifest /data/kaichen/data/MatchTime/manifests/valid.parquet \
    --base_url http://127.0.0.1:$((8100 + rank))/v1 \
    --chunk_turns 100 --rank $rank --world_size 8 \
    --output_dir $OUT >> $OUT/run.log 2>&1 < /dev/null &
done
