# StreamMind × OneVision-2

Streaming soccer commentary on top of the [StreamMind](https://arxiv.org/abs/2503.06220)
event-gated cognition framework, ported to a
[LLaVA-OneVision-2](https://arxiv.org/abs/2408.03326) backbone with
[MatchTime](https://arxiv.org/abs/2406.18530) style temporal supervision.

The pipeline keeps StreamMind's two-stage training (Stage-1: generation, Stage-2:
event gate) and its codec-based video representation, but swaps in an
OneVision-2 ViT + Qwen3 LLM as the base VLM and consumes precomputed per-segment
codec caches for fast, causal streaming training.

## 1. Environment

The image bundles PyTorch 2.9 + CUDA 12.9, `transformers>=5.7`, DeepSpeed,
`flash-attn` 2.8.3, `mamba-ssm`, `codec-video-prep`, and OpenCV. Everything is
installed against Python 3.12.

Build the image (recommended) from the repo root:

```bash
docker build -t streammind-ov:latest .
```

Run an interactive container with GPU access:

```bash
docker run --gpus all --shm-size=32g --rm -it \
    -v $PWD:/workspace/StreamMind \
    -w /workspace/StreamMind \
    streammind-ov:latest bash
```

If you prefer a local install, mirror the exact stack in `Dockerfile`; the pinned
versions have been the only ones we validated end to end.

## 2. Prepare data

We consume a parquet manifest that lists soccer halves (one row = one game +
half) plus their annotated commentary events. Each row is expanded into a set of
causal `(video_path, start, end)` segments before training/eval.

### 2.1 Directory layout

```
$MATCHTIME_ROOT/
    dataset/           # per-game commentary JSONs (upstream MatchTime)
    features_video/    # raw match videos (mkv) referenced by video_path
    manifests/
        train.parquet
        valid.parquet
    clips/             # per-segment MP4 subclips (built by cut_clips.py)
    codec_cache/       # per-segment codec canvases (built by gen_codec.py)
```

`MATCHTIME_ROOT` is the relpath base used to compute segment cache keys and
MUST match the value that training/eval sees.

### 2.2 Cut per-event subclips

`scripts/build_soccer_cache/cut_clips.py` reads the manifest, expands each row
into segments, and writes `ffmpeg -c copy` cuts under `--clip_dir`. Existing
non-empty clips are skipped, so it is safe to rerun.

```bash
python scripts/build_soccer_cache/cut_clips.py \
    --manifest       $MATCHTIME_ROOT/manifests/train.parquet \
    --matchtime_root $MATCHTIME_ROOT \
    --clip_dir       $MATCHTIME_ROOT/clips \
    --workers        16
```

Run the same command with `valid.parquet` to prepare the eval clips.

### 2.3 Generate the codec cache

`scripts/build_soccer_cache/gen_codec.py` invokes the OneVision-2 codec
processor on each clip and writes the canvases + patch-position tables under
`--codec_dir`. `--patch` and `--max_pixels` MUST match the values wired into the
trainer's cache-hit precheck (defaults match the shipped pipeline).

```bash
python scripts/build_soccer_cache/gen_codec.py \
    --manifest       $MATCHTIME_ROOT/manifests/train.parquet \
    --matchtime_root $MATCHTIME_ROOT \
    --clip_dir       $MATCHTIME_ROOT/clips \
    --codec_dir      $MATCHTIME_ROOT/codec_cache \
    --codec_module   /path/to/LLaVA-OneVision-2 \
    --workers        16
```

`--codec_module` points at any directory containing
`codec_video_processing_llava_onevision2.py`; the OneVision-2 checkpoint dir
works out of the box.

## 3. Train

Both stages share `ov_train/train.py`. The environment variables listed below
tell the loader where clips, codec caches, and manifests live; they must be
consistent with what was used at data-prep time.

```bash
export MATCHTIME_ROOT=/path/to/MatchTime
export STREAMMIND_CLIP_CACHE=$MATCHTIME_ROOT/clips
export ONLINE_CODEC_CACHE_DIR=$MATCHTIME_ROOT/codec_cache
export SOCCER_MANIFEST_DIR=$MATCHTIME_ROOT/manifests
```

`scripts/volcano_entrypoint.sh` is the reference launcher; it selects Stage-1
vs Stage-2 from `$STAGE` and passes the correct flags into `ov_train/train.py`.
It runs `torchrun` on a single node and expects an OneVision-2 backbone via
`$MODEL_PATH`.

### 3.1 Stage 1 — generation

Trains the projector + LLM on event-aligned commentary tokens under the causal
"no future frame" constraint.

```bash
STAGE=1 VIDEO_BACKEND=codec \
MODEL_PATH=/path/to/LLaVA-OneVision-2 \
OUTPUT_DIR=./outputs/stage1 \
LR=2e-5 \
DEEPSPEED_CONFIG=scripts/zero2.json \
bash scripts/volcano_entrypoint.sh
```

### 3.2 Stage 2 — event gate

Loads the Stage-1 checkpoint and trains only the classification head that
decides when the model should speak. Use the final Stage-1 checkpoint as
`$RESUME_FROM`.

```bash
STAGE=2 VIDEO_BACKEND=codec \
MODEL_PATH=/path/to/LLaVA-OneVision-2 \
RESUME_FROM=./outputs/stage1/checkpoint-final \
OUTPUT_DIR=./outputs/stage2 \
LR=2e-5 \
DEEPSPEED_CONFIG=scripts/zero2.json \
bash scripts/volcano_entrypoint.sh
```

Set `WANDB_API_KEY` to stream metrics to Weights & Biases; otherwise the
launcher falls back to `--report_to none`.

## 4. Evaluate

Both eval flavours share `ov_train/evaluate.py`. They reuse the same
`LazySupervisedDataset` as training so the eval distribution matches what the
model was trained on.

### 4.1 Event-gate metrics (`--eval_type cls`)

Reports the paper's TriggerAcc and TimeVal on the validation set:

```bash
python -m ov_train.evaluate \
    --model_path   /path/to/LLaVA-OneVision-2 \
    --resume_from  ./outputs/stage2/checkpoint-final \
    --eval_type    cls \
    --data_type    valid \
    --tolerance_frames 2
```

### 4.2 Caption metrics (`--eval_type llm`)

Dumps teacher-forced (pred, target) pairs into a CSV and reports PPL,
Correctness, and Fluency:

```bash
python -m ov_train.evaluate \
    --model_path   /path/to/LLaVA-OneVision-2 \
    --resume_from  ./outputs/stage1/checkpoint-final \
    --eval_type    llm \
    --data_type    valid \
    --caption_csv  ./outputs/stage1/eval_captions.csv
```

Once the CSV is dumped, compute BLEU/METEOR/ROUGE-L via the standalone scorer
(requires `pycocoevalcap` and a JRE for METEOR):

```bash
pip install pycocoevalcap
python -m ov_train.nlg_eval --csv ./outputs/stage1/eval_captions.csv
```

## 5. Citation

If this repository is useful, please cite the original StreamMind paper this
project builds upon:

```bibtex
@article{ding2025streammind,
  title={StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition},
  author={Ding, Xin and Wu, Hao and Yang, Yifan and Jiang, Shiqi and Bai, Donglin and Chen, Zhibo and Cao, Ting},
  journal={arXiv preprint arXiv:2503.06220},
  year={2025}
}
```

## 6. Acknowledgement

The event-gated streaming design, the two-stage generation / classifier
recipe, and the reference dataloader all come from
[**StreamMind**](https://github.com/xinding-sys/StreamMind). This repository
adapts that framework onto an OneVision-2 backbone with a codec-based data
pipeline.
