# Dockerfile for OneVision × StreamMind training
# Reproduces the exact env we built incrementally in plan tasks 0 + 7.
#
# Build:   docker build -t ov-stream:latest .
# Run:     docker run --gpus all --shm-size=32g \
#              -v /data/v-kaichen:/data/v-kaichen \
#              -w /data/v-kaichen/StreamMind \
#              ov-stream:latest \
#              bash scripts/finetune_ov_stage1_frames.sh

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

# ---- System deps ----
# ffmpeg 4.4+ required by codec backend (Ubuntu 22.04 ships 4.4.x)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv python3-pip \
        ffmpeg git wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    python -m pip install --upgrade pip

# ---- Torch 2.9 stack ----
RUN pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0

# ---- HF + video stack ----
RUN pip install \
        "transformers>=5.7.0" \
        "accelerate==1.*" \
        deepspeed==0.19.1 \
        decord==0.6.0 \
        pillow sentencepiece timm numpy==1.26.4

# ---- flash-attn 2.8.3 prebuilt wheel ----
# `pip install flash-attn==2.8.3` can fall back to source build on systems where
# pip's tmp dir lives on a different filesystem than ~/.cache/pip/wheels (the
# wheel move fails with "Invalid cross-device link"). Pull the exact wheel
# directly to avoid that.
RUN wget -q -O /tmp/flash_attn.whl \
        https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
    && pip install /tmp/flash_attn.whl \
    && rm /tmp/flash_attn.whl

# ---- Codec video backend ----
RUN pip install codec-video-prep==0.2.5 opencv-python==4.11.0.86

# ---- Sanity check ----
RUN python -c "import torch, transformers, flash_attn, decord, accelerate, deepspeed, codec_video_prep; \
print('torch', torch.__version__); \
print('transformers', transformers.__version__); \
print('flash_attn', flash_attn.__version__); \
print('decord', decord.__version__); \
print('deepspeed', deepspeed.__version__)"

WORKDIR /workspace
CMD ["bash"]
