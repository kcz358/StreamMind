#!/bin/bash
# Summariser models for the JoyAI 3-tier memory, co-located with the main servers.
cd /root/joyai_agent
mkdir -p logs
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
  setsid .venv/bin/vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --served-model-name summarizer --port $((8200 + gpu)) \
    --max-model-len 32768 --enable-prefix-caching --enable-chunked-prefill \
    --limit-mm-per-prompt '{"image":32,"video":0}' \
    --gpu-memory-utilization 0.10 \
    > logs/summarizer_$gpu.log 2>&1 < /dev/null &
done
