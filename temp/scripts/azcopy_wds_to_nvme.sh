#!/usr/bin/env bash
set -euo pipefail

SRC='https://mcgvisionflowsa.blob.core.windows.net/anxiang-data/llava-onevision-2/180s_streaming/?sv=2025-07-05&spr=https&st=2026-07-05T13%3A06%3A31Z&se=2026-07-12T13%3A06%3A00Z&skoid=4f6fae7d-48cb-4a3c-a0ec-a3614b8d4de0&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2026-07-05T13%3A06%3A31Z&ske=2026-07-12T13%3A06%3A00Z&sks=b&skv=2025-07-05&sr=c&sp=racwdxltf&sig=1kSxFoRgn5x8bIVJQnJhzqKxM%2FbwpPjUjuNz36WypBI%3D'
DST=/local_nvme/wds/180s_streaming

mkdir -p "$DST"

export AZCOPY_CONCURRENCY_VALUE=64
export AZCOPY_BUFFER_GB=32
export AZCOPY_LOG_LOCATION=/local_nvme/wds/azcopy_logs
export AZCOPY_JOB_PLAN_LOCATION=/local_nvme/wds/azcopy_plans
mkdir -p "$AZCOPY_LOG_LOCATION" "$AZCOPY_JOB_PLAN_LOCATION"

azcopy copy "$SRC" "$DST" \
  --recursive \
  --overwrite=ifSourceNewer \
  --check-md5=NoCheck \
  --log-level=INFO
