#!/bin/bash
# Launch one vLLM server per GPU for the JoyAI agent-system evaluation.
cd /root/joyai_agent
mkdir -p logs
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
  setsid .venv/bin/vllm serve jdopensource/JoyAI-VL-Interaction-Preview \
    --served-model-name joyai --port $((8100 + gpu)) \
    --max-model-len 65536 --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":128,"video":0}' \
    --gpu-memory-utilization 0.85 \
    > logs/server_$gpu.log 2>&1 < /dev/null &
done
