# OneVision × StreamMind Two-Stage Training Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train LLaVA-OneVision-2-8B-Instruct on SoccerNet/MatchTime using StreamMind's two-stage paradigm (Stage1: LLM learns *what* to say; Stage2: a small gate head learns *when* to say it), with both `frames` and `codec` video backends supported.

**Architecture:** A thin `StreamOneVision(nn.Module)` wraps `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and adds a 2-layer MLP gate head on the last hidden state. No patching of the remote modeling file. Dataset uses the bundled `AutoProcessor` for both backends; codec backend additionally pre-cuts 8-second clips with `ffmpeg -c copy` so cv-preinfer always sees a small file.

**Tech Stack:** torch==2.9.0, transformers>=5.7.0, flash-attn==2.8.3 (prebuilt wheel), decord, accelerate, deepspeed (reuse StreamMind's `scripts/zero2.json`), ffmpeg 6.x (system), codec-video-prep (codec backend only).

---

## Locked Decisions

- **Env:** reuse existing `/data/v-kaichen/StreamMind/.venv`. Do NOT create a new venv.
- **GPUs:** 4× A6000 48GB available. All smoke runs use 4-GPU `torchrun` + DeepSpeed zero2 (reuse `scripts/zero2.json`). Single-GPU is only a fallback if multi-GPU fails.
- **Model path:** `/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct`
- **Data root:** `/data/v-kaichen/azure_blob/data/MatchTime`
- **Clip cache:** `/data/v-kaichen/azure_blob/data/MatchTime/clips/` (blob, persistent, shared across runs). Each clip keyed by `(game_id, t_center)` → `.mp4`; created once per `(game, t)` and reused regardless of codec params.
- **Codec cache:** `/data/v-kaichen/azure_blob/data/MatchTime/codec_cache/` (set via `ONLINE_CODEC_CACHE_DIR` env). Keyed by `(clip_path, target_canvas, group_size)`.
- **Window:** 8 seconds before `t_center`, 16 effective frames per sample (both backends).
- **Codec config (tuned for short clip + ~16 effective frames):**
  `target_canvas=8, group_size=8, images_per_group=4, min_group_frames=4, max_pixels=150000`
  → `num_sampled_frames = (8/4)*8 = 16`
- **Backend switch:** CLI flag `--video_backend {frames,codec}` on train script, forwarded into dataset → processor.
- **Stage1 smoke first uses `frames`** (no extra deps, validates forward/backward/freeze). Stage1 codec smoke runs after frames smoke passes.
- **Gate head:** `Linear(4096, 4096) → GELU → Linear(4096, 2)` on `hidden_states[-1][:, gate_pos, :]`. CE with class weights `[0.15, 0.85]`.
- **Freeze logic:**
  - Stage 1 (`--stage 1`): `model.gate.requires_grad_(False)`, rest trainable.
  - Stage 2 (`--stage 2`): `model.base.requires_grad_(False)`, only `model.gate` trainable.
- **Data source for BOTH stages = MatchTime** (same caption JSONs + same SoccerNet videos). The only difference is sample construction:
  - **Stage 1 sample:** for each caption `(t_i, text_i)` → one positive. Loss = LM loss on assistant `text_i`. No gate labels emitted.
  - **Stage 2 sample:** for each caption `(t_i)` → one positive with `gate_label=1`. Additionally, for every positive, sample `K=3` negative timestamps from the "quiet" gaps between consecutive captions (must be ≥3s away from any caption); each negative gets `gate_label=0`. No assistant text (LM loss skipped). `gate_pos` = index of the last non-pad token in `input_ids`.

---

## Scope of This Plan

**In scope:** Environment install, wrapper, codec_utils, dataset (Stage1 + Stage2, both backends), train script (both stages), 10-step smoke on both stages × both backends, all running 4-GPU.

**Out of scope (separate future plan):** real (non-smoke) training, inference/streaming eval, multi-sample collator beyond batch=1 per device.

---

## File Structure

All new code lives in a dedicated top-level package `ov_train/` to avoid the legacy `streammind/` import chain (broken under transformers 5.x: `streammind/__init__.py` → `streammind/model/multimodal_projector/builder.py` imports the removed `TRANSFORMERS_CACHE` symbol; `data/datasets.py` needs `Levenshtein`). Do NOT touch any file under `streammind/` or `data/`.

| Path | Status | Responsibility |
|---|---|---|
| `ov_train/onevision_stream.py` | DONE | `StreamOneVision` wrapper + gate head |
| `ov_train/codec_utils.py` | NEW | `extract_subclip(src_video, t_start, dur, out_path)` — ffmpeg subprocess, idempotent |
| `ov_train/soccer_dataset.py` | NEW | `SoccerOneVisionDataset` + collator; reads MatchTime captions, calls processor for both backends |
| `ov_train/train.py` | NEW | Train entry: HF `Trainer`, args, model build, freeze logic |
| `scripts/finetune_ov_stage1_frames.sh` | NEW | Single-GPU 10-step smoke, frames backend |
| `scripts/finetune_ov_stage1_codec.sh` | NEW | Single-GPU 10-step smoke, codec backend |

No existing files are modified.

---

## Task 0: Install Environment (frames batch)

**Files:** none (env only).

- [ ] **Step 1:** Verify GPU + ffmpeg + venv exist.

  Run:
  ```bash
  nvidia-smi --query-gpu=index,memory.free --format=csv
  ffmpeg -version | head -1
  ls /data/v-kaichen/StreamMind/.venv/bin/python
  ```
  Expected: 4 GPUs listed with >40GB free each, ffmpeg 6.x, python binary exists.

- [ ] **Step 2:** Install torch 2.9 stack.

  Run:
  ```bash
  /data/v-kaichen/StreamMind/.venv/bin/pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
  ```
  Expected: completes without "No matching distribution" error.

- [ ] **Step 3:** Verify torch sees CUDA.

  Run:
  ```bash
  /data/v-kaichen/StreamMind/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
  ```
  Expected: `2.9.0 True 4`.

- [ ] **Step 4:** Install HF stack + decord.

  Run:
  ```bash
  /data/v-kaichen/StreamMind/.venv/bin/pip install "transformers>=5.7.0" accelerate==1.* decord==0.6.0 pillow sentencepiece
  ```
  Expected: completes; transformers shows 5.7.x or later.

- [ ] **Step 5:** Install flash-attn 2.8.3 (prebuilt wheel).

  Run:
  ```bash
  /data/v-kaichen/StreamMind/.venv/bin/pip install flash-attn==2.8.3 --no-build-isolation
  ```
  Expected: installs prebuilt wheel (no nvcc compile). If it tries to build from source, abort and report to user.

- [ ] **Step 6:** Smoke-import everything.

  Run:
  ```bash
  /data/v-kaichen/StreamMind/.venv/bin/python -c "
  import torch, transformers, flash_attn, decord, accelerate
  print('torch', torch.__version__)
  print('transformers', transformers.__version__)
  print('flash_attn', flash_attn.__version__)
  print('decord', decord.__version__)
  "
  ```
  Expected: all four print versions with no ImportError.

---

## Task 1: Load Model Smoke Test (sanity, before writing wrapper)

**Files:** none (one-off script, not committed).

- [ ] **Step 1:** Confirm the OneVision checkpoint loads via Auto* and reports the expected hidden size.

  Run:
  ```bash
  /data/v-kaichen/StreamMind/.venv/bin/python -c "
  import torch
  from transformers import AutoModelForCausalLM, AutoProcessor
  P = '/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct'
  proc = AutoProcessor.from_pretrained(P, trust_remote_code=True)
  print('processor OK:', type(proc).__name__)
  m = AutoModelForCausalLM.from_pretrained(P, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map='cuda:0')
  print('model OK:', type(m).__name__, 'hidden_size=', m.config.text_config.hidden_size)
  print('VRAM GB:', torch.cuda.memory_allocated()/1e9)
  "
  ```
  Expected: `processor OK: LlavaOnevision2Processor`, `model OK: LlavaOnevision2ForConditionalGeneration hidden_size= 4096`, VRAM ~16-17 GB. **If this fails, STOP — debug before proceeding.**

---

## Task 2: Write `streammind/model/onevision_stream.py`

**Files:**
- Create: `streammind/model/onevision_stream.py`

- [ ] **Step 1:** Write the wrapper.

  ```python
  # streammind/model/onevision_stream.py
  """StreamMind-style wrapper around LLaVA-OneVision-2 for two-stage training.

  Stage 1: train the underlying CausalLM on video captions (gate frozen).
  Stage 2: freeze base, train only the gate head on silence/response labels.
  """
  from __future__ import annotations

  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from transformers import AutoModelForCausalLM
  from transformers.modeling_outputs import CausalLMOutputWithPast


  class StreamOneVision(nn.Module):
      def __init__(self, model_name_or_path: str, gate_class_weights=(0.15, 0.85)):
          super().__init__()
          self.base = AutoModelForCausalLM.from_pretrained(
              model_name_or_path,
              trust_remote_code=True,
              torch_dtype=torch.bfloat16,
              attn_implementation="flash_attention_2",
          )
          hidden = self.base.config.text_config.hidden_size  # 4096 for Qwen3-8B
          self.gate = nn.Sequential(
              nn.Linear(hidden, hidden),
              nn.GELU(),
              nn.Linear(hidden, 2),
          )
          # gate stays in fp32 for numerical stability; small enough not to matter
          self.register_buffer(
              "_gate_w",
              torch.tensor(gate_class_weights, dtype=torch.float32),
              persistent=False,
          )

      def gradient_checkpointing_enable(self, **kwargs):
          self.base.gradient_checkpointing_enable(**kwargs)

      def forward(
          self,
          gate_label: torch.LongTensor | None = None,
          gate_pos: torch.LongTensor | None = None,
          **inputs,
      ):
          need_h = gate_label is not None
          out = self.base(**inputs, output_hidden_states=need_h, return_dict=True)
          loss = out.loss

          if need_h:
              h = out.hidden_states[-1]                              # [B, T, H]
              b_idx = torch.arange(h.size(0), device=h.device)
              g_logits = self.gate(h[b_idx, gate_pos].float())       # [B, 2]
              gate_loss = F.cross_entropy(
                  g_logits, gate_label, weight=self._gate_w.to(g_logits.device)
              )
              loss = gate_loss if loss is None else loss + gate_loss

          return CausalLMOutputWithPast(
              loss=loss,
              logits=out.logits,
              past_key_values=getattr(out, "past_key_values", None),
              hidden_states=out.hidden_states if need_h else None,
          )
  ```

- [ ] **Step 2:** Smoke-import the wrapper and construct it.

  Run:
  ```bash
  /data/v-kaichen/StreamMind/.venv/bin/python -c "
  from streammind.model.onevision_stream import StreamOneVision
  m = StreamOneVision('/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct').cuda()
  print('OK params:', sum(p.numel() for p in m.parameters())/1e9, 'B')
  print('gate params:', sum(p.numel() for p in m.gate.parameters())/1e6, 'M')
  "
  ```
  Expected: ~8B total, ~33M gate, no errors.

- [ ] **Step 3:** Commit.

  ```bash
  cd /data/v-kaichen/StreamMind
  git add streammind/model/onevision_stream.py
  git commit -m "feat: add StreamOneVision wrapper with gate head"
  ```

---

## Task 3: Write `data/codec_utils.py`

**Files:**
- Create: `data/codec_utils.py`

- [ ] **Step 1:** Write the subclip extractor.

  ```python
  # data/codec_utils.py
  """Idempotent 8-second subclip extraction for codec backend.

  ffmpeg -c copy is fast (no re-encode) but only seeks to nearest keyframe;
  that's acceptable here because (a) MatchTime timestamps have ~1s tolerance
  already, (b) cv-preinfer operates on whatever stream we give it.
  """
  from __future__ import annotations

  import os
  import shutil
  import subprocess
  import tempfile
  from pathlib import Path


  def extract_subclip(
      src_video: str | Path,
      t_start: float,
      duration: float,
      out_path: str | Path,
      timeout: int = 120,
  ) -> Path:
      """Cut [t_start, t_start+duration] from src_video → out_path. Idempotent.

      Returns the resolved out_path. If out_path already exists and is non-empty,
      returns immediately. Writes atomically via a tmp file in the same dir.
      """
      out_path = Path(out_path)
      if out_path.exists() and out_path.stat().st_size > 0:
          return out_path

      out_path.parent.mkdir(parents=True, exist_ok=True)
      tmp_fd, tmp_name = tempfile.mkstemp(
          suffix=out_path.suffix, prefix=".tmp_", dir=out_path.parent
      )
      os.close(tmp_fd)
      tmp_path = Path(tmp_name)

      cmd = [
          "ffmpeg", "-y", "-loglevel", "error",
          "-ss", f"{max(0.0, t_start):.3f}",
          "-t", f"{duration:.3f}",
          "-i", str(src_video),
          "-c", "copy",
          "-avoid_negative_ts", "make_zero",
          str(tmp_path),
      ]
      try:
          subprocess.run(cmd, check=True, timeout=timeout, capture_output=True)
      except subprocess.CalledProcessError as e:
          tmp_path.unlink(missing_ok=True)
          raise RuntimeError(
              f"ffmpeg failed for {src_video} @ {t_start}s: {e.stderr.decode(errors='ignore')[:500]}"
          ) from e

      # atomic rename
      shutil.move(str(tmp_path), str(out_path))
      return out_path
  ```

- [ ] **Step 2:** Smoke-test the extractor against one real SoccerNet file.

  Run:
  ```bash
  /data/v-kaichen/StreamMind/.venv/bin/python -c "
  from data.codec_utils import extract_subclip
  from pathlib import Path
  # Pick any downloaded video. Adjust the glob if dir layout differs.
  import glob
  vids = sorted(glob.glob('/data/v-kaichen/azure_blob/data/MatchTime/features_video/*/*/1_224p.mkv'))
  assert vids, 'no SoccerNet videos found — check data path'
  out = extract_subclip(vids[0], 600.0, 8.0, '/tmp/_test_clip.mp4')
  print('clip:', out, 'size:', Path(out).stat().st_size)
  "
  ```
  Expected: prints path and a non-zero size (typically 1–5 MB).

- [ ] **Step 3:** Commit.

  ```bash
  cd /data/v-kaichen/StreamMind
  git add data/codec_utils.py
  git commit -m "feat: add idempotent ffmpeg subclip extractor for codec backend"
  ```

---

## Task 4: Write `data/soccer_onevision.py` (Stage1, both backends)

**Files:**
- Create: `data/soccer_onevision.py`

This task is the largest. Read it fully before starting.

- [ ] **Step 1:** Skim StreamMind's existing video↔caption pairing helpers so the new dataset uses the same file conventions.

  Read: `data/soccer_data.py` (functions `trans_video_2_json`, `find_video_files`, `extract_video_half`). Confirm:
  - Caption JSON path: `dataset/MatchTime/{split}/{league_season}/{game}/Labels-caption.json`
  - Video path: `features_video/{league_season}/{game}/{1|2}_224p.mkv` (half = 1 or 2)
  - Each caption entry has `{"gameTime": "<half> - MM:SS", "anonymized": "..."}`

- [ ] **Step 2:** Write the dataset and collator.

  ```python
  # data/soccer_onevision.py
  """Stage-1 dataset for OneVision two-stage training on SoccerNet+MatchTime.

  One sample = (game_video, t_center, caption). Produces processor-ready tensors
  with labels masked so loss is only computed on the assistant caption tokens.

  Supports two backends, switched at construction time:
    - "frames": decord reads num_frames uniformly from [t-window, t]
    - "codec":  pre-cut an 8s clip via ffmpeg, feed clip path to processor
                with video_backend="codec" + tuned codec_config.
  """
  from __future__ import annotations

  import json
  import os
  from dataclasses import dataclass
  from pathlib import Path
  from typing import Any

  import numpy as np
  import torch
  from torch.utils.data import Dataset

  from data.codec_utils import extract_subclip

  DATA_ROOT = Path("/data/v-kaichen/azure_blob/data/MatchTime")
  CAPTION_ROOT = DATA_ROOT / "dataset" / "MatchTime"
  VIDEO_ROOT = DATA_ROOT / "features_video"
  CLIP_CACHE = DATA_ROOT / "clips"

  CODEC_CONFIG = {
      "target_canvas": 8,
      "group_size": 8,
      "images_per_group": 4,
      "min_group_frames": 4,
  }
  CODEC_MAX_PIXELS = 150_000

  SYSTEM_PROMPT = "You are a live soccer commentator."
  USER_PROMPT = "Describe what just happened in this clip in one sentence."


  def _parse_game_time(gt: str) -> tuple[int, float]:
      """'1 - 23:45' -> (1, 1425.0)"""
      half, mmss = gt.split(" - ")
      mm, ss = mmss.split(":")
      return int(half), int(mm) * 60 + int(ss)


  def _list_samples(split: str) -> list[dict]:
      out = []
      split_dir = CAPTION_ROOT / split
      for cap_json in split_dir.glob("*/*/Labels-caption.json"):
          league_season = cap_json.parent.parent.name
          game = cap_json.parent.name
          try:
              data = json.loads(cap_json.read_text())
          except Exception:
              continue
          for ann in data.get("annotations", []):
              text = ann.get("anonymized") or ann.get("description")
              gt = ann.get("gameTime")
              if not text or not gt:
                  continue
              try:
                  half, t = _parse_game_time(gt)
              except Exception:
                  continue
              if half not in (1, 2) or t < 8.0:
                  continue  # need 8s of context before t
              video = VIDEO_ROOT / league_season / game / f"{half}_224p.mkv"
              if not video.exists():
                  continue
              out.append({
                  "video": str(video),
                  "t": float(t),
                  "caption": text.strip(),
                  "game_id": f"{league_season}__{game}__h{half}",
              })
      return out


      class SoccerOneVisionDataset(Dataset):
          def __init__(
              self,
              processor,
              split: str = "train",
              video_backend: str = "frames",
              num_frames: int = 16,
              window_seconds: float = 8.0,
              max_samples: int | None = None,
              stage: int = 1,
              neg_per_pos: int = 3,
              neg_min_gap: float = 3.0,
              seed: int = 0,
          ):
              assert video_backend in ("frames", "codec")
              assert stage in (1, 2)
              self.processor = processor
              self.video_backend = video_backend
              self.num_frames = num_frames
              self.window = window_seconds
              self.stage = stage
              positives = _list_samples(split)
              if stage == 1:
                  self.samples = positives
              else:
                  self.samples = self._build_stage2(positives, neg_per_pos, neg_min_gap, seed)
              if max_samples is not None:
                  self.samples = self.samples[:max_samples]
              if video_backend == "codec":
                  CLIP_CACHE.mkdir(parents=True, exist_ok=True)
                  os.environ.setdefault(
                      "ONLINE_CODEC_CACHE_DIR", str(DATA_ROOT / "codec_cache")
                  )

          @staticmethod
          def _build_stage2(positives, neg_per_pos, min_gap, seed):
              """Expand positives with K negatives sampled from quiet gaps per game_id."""
              rng = np.random.default_rng(seed)
              by_game: dict[str, list[dict]] = {}
              for p in positives:
                  by_game.setdefault(p["game_id"], []).append(p)
              out = []
              for game_id, plist in by_game.items():
                  plist = sorted(plist, key=lambda x: x["t"])
                  ts = [p["t"] for p in plist]
                  v = plist[0]["video"]
                  # positives → gate_label=1
                  for p in plist:
                      out.append({**p, "gate_label": 1})
                  # negatives: pick times in [min(ts)+8, max(ts)] that are >= min_gap from any t
                  if len(ts) < 2:
                      continue
                  lo, hi = ts[0] + 8.0, ts[-1]
                  if hi - lo < 4 * min_gap:
                      continue
                  ts_arr = np.array(ts)
                  need = neg_per_pos * len(plist)
                  picked = 0
                  attempts = 0
                  while picked < need and attempts < need * 20:
                      attempts += 1
                      cand = float(rng.uniform(lo, hi))
                      if np.min(np.abs(ts_arr - cand)) < min_gap:
                          continue
                      out.append({
                          "video": v, "t": cand, "caption": "",
                          "game_id": game_id, "gate_label": 0,
                      })
                      picked += 1
              rng.shuffle(out)
              return out

          def __len__(self):
              return len(self.samples)

      # ------------------------------------------------------------------
      def _load_frames_decord(self, video_path: str, t_end: float) -> np.ndarray:
          import decord
          decord.bridge.set_bridge("native")
          vr = decord.VideoReader(video_path, num_threads=1)
          fps = float(vr.get_avg_fps()) or 25.0
          t_start = max(0.0, t_end - self.window)
          f_start = int(round(t_start * fps))
          f_end = min(len(vr) - 1, int(round(t_end * fps)))
          if f_end - f_start < self.num_frames:
              f_end = min(len(vr) - 1, f_start + self.num_frames)
          idx = np.linspace(f_start, f_end, self.num_frames).astype(int)
          frames = vr.get_batch(idx).asnumpy()  # [N, H, W, 3] uint8
          return frames

      # ------------------------------------------------------------------
      def _build_chat(self, caption: str) -> list[dict]:
          return [
              {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
              {"role": "user", "content": [
                  {"type": "video"},
                  {"type": "text", "text": USER_PROMPT},
              ]},
              {"role": "assistant", "content": [{"type": "text", "text": caption}]},
          ]

      def _mask_prompt(self, full_ids: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
          """Set labels[:len(prompt_ids)] = -100 so only assistant tokens get loss."""
          labels = full_ids.clone()
          n = prompt_ids.size(-1)
          labels[..., :n] = -100
          return labels

      # ------------------------------------------------------------------
      def __getitem__(self, i):
          s = self.samples[i]

          if self.stage == 1:
              msgs = self._build_chat(s["caption"])
              text_full = self.processor.apply_chat_template(
                  msgs, tokenize=False, add_generation_prompt=False
              )
              text_prompt = self.processor.apply_chat_template(
                  msgs[:-1], tokenize=False, add_generation_prompt=True
              )
          else:
              # Stage 2: no assistant turn at all; we only need the prompt context.
              msgs_prompt = [
                  {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                  {"role": "user", "content": [
                      {"type": "video"},
                      {"type": "text", "text": USER_PROMPT},
                  ]},
              ]
              text_full = self.processor.apply_chat_template(
                  msgs_prompt, tokenize=False, add_generation_prompt=True
              )
              text_prompt = text_full  # all tokens masked from LM loss

          if self.video_backend == "frames":
              frames = self._load_frames_decord(s["video"], s["t"])
              video_kwargs = {"videos": [frames], "num_frames": self.num_frames}
          else:  # codec
              t_start = max(0.0, s["t"] - self.window)
              clip = CLIP_CACHE / f"{s['game_id']}__t{int(s['t']*100):08d}.mp4"
              extract_subclip(s["video"], t_start, self.window, clip)
              video_kwargs = {
                  "videos": [str(clip)],
                  "video_backend": "codec",
                  "codec_config": CODEC_CONFIG,
                  "max_pixels": CODEC_MAX_PIXELS,
              }

          enc_full = self.processor(
              text=[text_full], return_tensors="pt", padding=False, **video_kwargs
          )
          full_ids = enc_full["input_ids"][0]

          if self.stage == 1:
              enc_prompt = self.processor(
                  text=[text_prompt], return_tensors="pt", padding=False, **video_kwargs
              )
              prompt_ids = enc_prompt["input_ids"][0]
              labels = self._mask_prompt(full_ids, prompt_ids)
          else:
              labels = torch.full_like(full_ids, -100)  # no LM loss in stage 2

          item = {
              "input_ids": full_ids,
              "attention_mask": enc_full["attention_mask"][0],
              "labels": labels,
          }
          if self.stage == 2:
              item["gate_label"] = torch.tensor(s["gate_label"], dtype=torch.long)
              item["gate_pos"] = torch.tensor(full_ids.size(0) - 1, dtype=torch.long)

          # Forward vision tensors verbatim; collator handles batch=1 case.
          for k in ("pixel_values", "image_grid_thw", "patch_positions",
                    "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"):
              if k in enc_full:
                  item[k] = enc_full[k]
          return item


  @dataclass
  class OneVisionCollator:
      pad_token_id: int

      def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
          # MVP: batch size 1 only. Multi-sample padding deferred until needed.
          assert len(batch) == 1, "Stage1 smoke uses per_device_batch_size=1"
          item = batch[0]
          out = {
              "input_ids": item["input_ids"].unsqueeze(0),
              "attention_mask": item["attention_mask"].unsqueeze(0),
              "labels": item["labels"].unsqueeze(0),
          }
          if "gate_label" in item:
              out["gate_label"] = item["gate_label"].unsqueeze(0)
              out["gate_pos"] = item["gate_pos"].unsqueeze(0)
          for k in ("pixel_values", "image_grid_thw", "patch_positions",
                    "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"):
              if k in item:
                  out[k] = item[k]  # already batched/2D from processor
          return out
  ```

- [ ] **Step 3:** Smoke-test dataset (frames backend, 1 sample, no model).

  Run:
  ```bash
  /data/v-kaichen/StreamMind/.venv/bin/python -c "
  from transformers import AutoProcessor
  from data.soccer_onevision import SoccerOneVisionDataset
  P = '/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct'
  proc = AutoProcessor.from_pretrained(P, trust_remote_code=True)
  ds = SoccerOneVisionDataset(proc, split='valid', video_backend='frames', max_samples=2)
  print('len', len(ds))
  s = ds[0]
  print({k: (tuple(v.shape) if hasattr(v,'shape') else v) for k,v in s.items()})
  print('label tail (assistant tokens):', s['labels'][s['labels']!=-100][:20].tolist())
  "
  ```
  Expected: prints shapes for `input_ids`, `pixel_values`, etc.; the label tail contains real token ids (not -100), confirming masking worked.

- [ ] **Step 4:** Commit.

  ```bash
  cd /data/v-kaichen/StreamMind
  git add data/soccer_onevision.py
  git commit -m "feat: SoccerOneVision dataset with frames+codec backends (stage1)"
  ```

---

## Task 5: Write `streammind/train_onevision.py`

**Files:**
- Create: `streammind/train_onevision.py`

- [ ] **Step 1:** Write the training entry.

  ```python
  # streammind/train_onevision.py
  """Two-stage trainer for OneVision-based StreamMind.

  Stage 1: train base CausalLM on caption generation; gate frozen.
  Stage 2: freeze base, train only the gate head. (Dataset for stage 2 not
  built yet — this script only structures the freeze logic.)
  """
  from __future__ import annotations

  import os
  import sys
  from dataclasses import dataclass, field

  import torch
  import transformers
  from transformers import AutoProcessor, HfArgumentParser, Trainer, TrainingArguments

  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

  from data.soccer_onevision import OneVisionCollator, SoccerOneVisionDataset
  from streammind.model.onevision_stream import StreamOneVision


  @dataclass
  class ModelArgs:
      model_name_or_path: str = field(
          default="/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct"
      )


  @dataclass
  class DataArgs:
      split: str = "train"
      video_backend: str = "frames"  # "frames" or "codec"
      num_frames: int = 16
      window_seconds: float = 8.0
      max_samples: int | None = None  # cap for smoke testing


  @dataclass
  class StageArgs:
      stage: int = 1  # 1 = train LLM, 2 = train gate


  def apply_stage_freeze(model: StreamOneVision, stage: int):
      if stage == 1:
          for p in model.gate.parameters():
              p.requires_grad = False
          # base fully trainable (subject to gradient_checkpointing etc.)
      elif stage == 2:
          for p in model.base.parameters():
              p.requires_grad = False
          for p in model.gate.parameters():
              p.requires_grad = True
      else:
          raise ValueError(f"unknown stage {stage}")

      trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
      total = sum(p.numel() for p in model.parameters())
      print(f"[stage {stage}] trainable {trainable/1e6:.1f}M / total {total/1e9:.2f}B")


  def main():
      parser = HfArgumentParser((ModelArgs, DataArgs, StageArgs, TrainingArguments))
      model_args, data_args, stage_args, training_args = parser.parse_args_into_dataclasses()

      processor = AutoProcessor.from_pretrained(
          model_args.model_name_or_path, trust_remote_code=True
      )
      model = StreamOneVision(model_args.model_name_or_path)
      apply_stage_freeze(model, stage_args.stage)

      if training_args.gradient_checkpointing:
          model.gradient_checkpointing_enable(
              gradient_checkpointing_kwargs={"use_reentrant": False}
          )

      train_ds = SoccerOneVisionDataset(
          processor,
          split=data_args.split,
          video_backend=data_args.video_backend,
          num_frames=data_args.num_frames,
          window_seconds=data_args.window_seconds,
          max_samples=data_args.max_samples,
          stage=stage_args.stage,
      )
      collator = OneVisionCollator(pad_token_id=processor.tokenizer.pad_token_id or 0)

      trainer = Trainer(
          model=model,
          args=training_args,
          train_dataset=train_ds,
          data_collator=collator,
      )
      trainer.train()


  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 2:** Sanity-import the script (no run).

  Run:
  ```bash
  /data/v-kaichen/StreamMind/.venv/bin/python -c "import streammind.train_onevision; print('OK')"
  ```
  Expected: `OK`, no ImportError.

- [ ] **Step 3:** Commit.

  ```bash
  cd /data/v-kaichen/StreamMind
  git add streammind/train_onevision.py
  git commit -m "feat: train_onevision entry with two-stage freeze"
  ```

---

## Task 6: Stage1 Frames Smoke (10 steps, 4 GPU)

**Files:**
- Create: `scripts/finetune_ov_stage1_frames.sh`

- [ ] **Step 1:** Write the smoke script.

  ```bash
  #!/bin/bash
  # scripts/finetune_ov_stage1_frames.sh
  set -e
  cd "$(dirname "$0")/.."

  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export CUDA_VISIBLE_DEVICES=0,1,2,3

  .venv/bin/torchrun --nproc_per_node 4 --master_port 16667 \
      streammind/train_onevision.py \
      --stage 1 \
      --video_backend frames \
      --num_frames 16 \
      --split train \
      --max_samples 64 \
      --model_name_or_path /data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct \
      --output_dir output/ov_stage1_frames_smoke \
      --deepspeed scripts/zero2.json \
      --max_steps 10 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --learning_rate 2e-5 \
      --bf16 True \
      --tf32 True \
      --gradient_checkpointing True \
      --logging_steps 1 \
      --save_strategy no \
      --report_to none \
      --dataloader_num_workers 2 \
      --remove_unused_columns False
  ```

- [ ] **Step 2:** Make executable and run.

  Run:
  ```bash
  chmod +x scripts/finetune_ov_stage1_frames.sh
  bash scripts/finetune_ov_stage1_frames.sh 2>&1 | tee /tmp/ov_stage1_frames_smoke.log
  ```
  Expected:
  - 10 lines of `{'loss': X.XX, ...}` printed.
  - Loss is a finite number (not nan/inf).
  - No OOM. **If OOM, report VRAM peak from `nvidia-smi` and stop.**

- [ ] **Step 3:** Verify and report VRAM peak.

  Run:
  ```bash
  grep -E "'loss':" /tmp/ov_stage1_frames_smoke.log | head -10
  ```
  Expected: 10 loss lines, all finite. Also report (manually) the peak VRAM observed during the run via a separate `nvidia-smi` watch session, OR by parsing `torch.cuda.max_memory_allocated` printed at end (not added to script — read from `nvidia-smi dmon` if needed).

- [ ] **Step 4:** Commit.

  ```bash
  cd /data/v-kaichen/StreamMind
  git add scripts/finetune_ov_stage1_frames.sh
      git commit -m "feat: stage1 frames smoke script (10 steps, 4 GPU + zero2)"
  ```

---

## Task 7: Install Codec Backend Deps

**Files:** none.

- [ ] **Step 1:** Install codec deps.

  Run:
  ```bash
  /data/v-kaichen/StreamMind/.venv/bin/pip install codec-video-prep opencv-python
  ```
  Expected: completes; provides `cv-preinfer` CLI on PATH (inside the venv's `bin/`).

- [ ] **Step 2:** Verify CLI is reachable.

  Run:
  ```bash
  /data/v-kaichen/StreamMind/.venv/bin/cv-preinfer --help | head -5
  ```
  Expected: usage text. If not found, report which package failed.

---

## Task 8: Stage1 Codec Smoke (10 steps, 4 GPU)

**Files:**
- Create: `scripts/finetune_ov_stage1_codec.sh`

- [ ] **Step 1:** Write the codec smoke script.

  ```bash
  #!/bin/bash
  # scripts/finetune_ov_stage1_codec.sh
  set -e
  cd "$(dirname "$0")/.."

  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export ONLINE_CODEC_CACHE_DIR=/data/v-kaichen/azure_blob/data/MatchTime/codec_cache
  export PATH="$PWD/.venv/bin:$PATH"

  .venv/bin/torchrun --nproc_per_node 4 --master_port 16668 \
      streammind/train_onevision.py \
      --stage 1 \
      --video_backend codec \
      --num_frames 16 \
      --split train \
      --max_samples 64 \
      --model_name_or_path /data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct \
      --output_dir output/ov_stage1_codec_smoke \
      --deepspeed scripts/zero2.json \
      --max_steps 10 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --learning_rate 2e-5 \
      --bf16 True \
      --tf32 True \
      --gradient_checkpointing True \
      --logging_steps 1 \
      --save_strategy no \
      --report_to none \
      --dataloader_num_workers 0 \
      --remove_unused_columns False
  ```

  Note: `dataloader_num_workers=0` for codec smoke so cv-preinfer errors surface in the main process. Raise later for real training.

- [ ] **Step 2:** Run and observe.

  Run:
  ```bash
  chmod +x scripts/finetune_ov_stage1_codec.sh
  bash scripts/finetune_ov_stage1_codec.sh 2>&1 | tee /tmp/ov_stage1_codec_smoke.log
  ```
  Expected:
  - First sample is slow (clip cut + cv-preinfer first run).
  - 10 loss lines printed, all finite.
  - `/data/v-kaichen/azure_blob/data/MatchTime/clips/` contains new `.mp4` files.
  - `/data/v-kaichen/azure_blob/data/MatchTime/codec_cache/` contains canvas subdirectories.
  - **If OOM:** report VRAM and try `--num_frames 8` or lower `max_pixels` in `CODEC_MAX_PIXELS`.
  - **If cv-preinfer error:** capture full stderr and stop.

- [ ] **Step 3:** Verify.

  Run:
  ```bash
  grep -E "'loss':" /tmp/ov_stage1_codec_smoke.log | head -10
  ls /data/v-kaichen/azure_blob/data/MatchTime/clips/ | head -5
  ls /data/v-kaichen/azure_blob/data/MatchTime/codec_cache/ | head -5
  ```
  Expected: 10 loss lines; clips dir non-empty; codec_cache dir non-empty.

- [ ] **Step 4:** Commit.

  ```bash
  cd /data/v-kaichen/StreamMind
  git add scripts/finetune_ov_stage1_codec.sh
  git commit -m "feat: stage1 codec smoke script with clip+codec caching"
  ```

---

## Task 9: Stage2 Frames Smoke (10 steps, 4 GPU)

**Files:**
- Create: `scripts/finetune_ov_stage2_frames.sh`

- [ ] **Step 1:** Write the smoke script. Same as Task 6 but `--stage 2`.

  ```bash
  #!/bin/bash
  # scripts/finetune_ov_stage2_frames.sh
  set -e
  cd "$(dirname "$0")/.."

  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export CUDA_VISIBLE_DEVICES=0,1,2,3

  .venv/bin/torchrun --nproc_per_node 4 --master_port 16669 \
      streammind/train_onevision.py \
      --stage 2 \
      --video_backend frames \
      --num_frames 16 \
      --split train \
      --max_samples 64 \
      --model_name_or_path /data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct \
      --output_dir output/ov_stage2_frames_smoke \
      --deepspeed scripts/zero2.json \
      --max_steps 10 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --learning_rate 1e-4 \
      --bf16 True \
      --tf32 True \
      --gradient_checkpointing True \
      --logging_steps 1 \
      --save_strategy no \
      --report_to none \
      --dataloader_num_workers 2 \
      --remove_unused_columns False
  ```

- [ ] **Step 2:** Run and observe.

  Run:
  ```bash
  chmod +x scripts/finetune_ov_stage2_frames.sh
  bash scripts/finetune_ov_stage2_frames.sh 2>&1 | tee /tmp/ov_stage2_frames_smoke.log
  ```
  Expected:
  - At startup, the freeze line prints `[stage 2] trainable ~33.0M / total ~8.0B`.
  - 10 loss lines printed, all finite. Loss should be roughly CE on 2-class (~0.69 at init, drifting).
  - No OOM. (Stage 2 still needs full base forward, so VRAM ≈ stage 1.)

- [ ] **Step 3:** Verify.

  Run:
  ```bash
  grep -E "trainable|'loss':" /tmp/ov_stage2_frames_smoke.log | head -15
  ```
  Expected: one `trainable ~33.0M / total ~8.0B` line + 10 loss lines.

- [ ] **Step 4:** Commit.

  ```bash
  cd /data/v-kaichen/StreamMind
  git add scripts/finetune_ov_stage2_frames.sh
  git commit -m "feat: stage2 frames smoke (gate-only training)"
  ```

---

## Task 10: Stage2 Codec Smoke (10 steps, 4 GPU)

**Files:**
- Create: `scripts/finetune_ov_stage2_codec.sh`

- [ ] **Step 1:** Write the smoke script.

  ```bash
  #!/bin/bash
  # scripts/finetune_ov_stage2_codec.sh
  set -e
  cd "$(dirname "$0")/.."

  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export ONLINE_CODEC_CACHE_DIR=/data/v-kaichen/azure_blob/data/MatchTime/codec_cache
  export PATH="$PWD/.venv/bin:$PATH"

  .venv/bin/torchrun --nproc_per_node 4 --master_port 16670 \
      streammind/train_onevision.py \
      --stage 2 \
      --video_backend codec \
      --num_frames 16 \
      --split train \
      --max_samples 64 \
      --model_name_or_path /data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct \
      --output_dir output/ov_stage2_codec_smoke \
      --deepspeed scripts/zero2.json \
      --max_steps 10 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --learning_rate 1e-4 \
      --bf16 True \
      --tf32 True \
      --gradient_checkpointing True \
      --logging_steps 1 \
      --save_strategy no \
      --report_to none \
      --dataloader_num_workers 0 \
      --remove_unused_columns False
  ```

- [ ] **Step 2:** Run.

  Run:
  ```bash
  chmod +x scripts/finetune_ov_stage2_codec.sh
  bash scripts/finetune_ov_stage2_codec.sh 2>&1 | tee /tmp/ov_stage2_codec_smoke.log
  ```
  Expected: same as Task 9 step 2, with codec caches reused from Task 8.

- [ ] **Step 3:** Verify.

  Run:
  ```bash
  grep -E "trainable|'loss':" /tmp/ov_stage2_codec_smoke.log | head -15
  ```
  Expected: 1 freeze line + 10 loss lines.

- [ ] **Step 4:** Commit.

  ```bash
  cd /data/v-kaichen/StreamMind
  git add scripts/finetune_ov_stage2_codec.sh
  git commit -m "feat: stage2 codec smoke (gate-only training)"
  ```

---

## Done Criteria

- [ ] Task 0–6 complete: stage1 frames smoke runs 10 steps, finite loss.
- [ ] Task 7–8 complete: stage1 codec smoke runs 10 steps, finite loss, caches populated.
- [ ] Task 9 complete: stage2 frames smoke runs 10 steps, gate-only trainable count printed, finite loss.
- [ ] Task 10 complete: stage2 codec smoke runs 10 steps, finite loss.
- [ ] VRAM peak reported for all four runs.
- [ ] All commits landed on the current branch.

## Out-of-Scope Follow-ups (separate plan)

- Real (non-smoke) training: full MatchTime, multi-epoch, checkpoint save (`_save` override for `StreamOneVision`).
- Inference / streaming eval script (gate-driven generation loop).
- Multi-sample collator (left/right padding for `input_ids` and vision tensors); current collator is batch=1 per device.

---

## Self-Review Notes

- **Spec coverage:** wrapper ✓, codec utils ✓, dataset (both backends, both stages) ✓, train script (both stages) ✓, stage1 frames smoke ✓, stage1 codec smoke ✓, stage2 frames smoke ✓, stage2 codec smoke ✓, env install ✓, 4-GPU + DeepSpeed ✓.
- **Placeholder scan:** no TBD/TODO; all code blocks complete.
- **Type consistency:** `StreamOneVision`, `SoccerOneVisionDataset`, `OneVisionCollator`, `extract_subclip`, `apply_stage_freeze` referenced consistently across tasks.
- **Known soft spots:**
  1. `Trainer` may complain that the wrapper isn't a `PreTrainedModel` (no `.save_pretrained`). With `save_strategy=no` this is fine for smoke; full training needs a `_save` override or `save_safetensors=False`. Flagged for follow-up.
  2. `processor` is called twice in `__getitem__` (full vs prompt-only) — wasteful but simple. Optimize after correctness is confirmed.
  3. Codec backend assumes cv-preinfer's `<X.X seconds>` text tags fit the chat template; if processor fails to expand, fall back to inspecting `chat_template.jinja` and adjusting `_build_chat`. Captured as a risk; not pre-emptively coded.
