# ov_train/soccer_dataset.py
"""Stage-1 + Stage-2 dataset for OneVision two-stage training on SoccerNet+MatchTime.

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

from ov_train.codec_utils import extract_subclip

DATA_ROOT = Path(
    os.environ.get("MATCHTIME_ROOT", "/data/v-kaichen/azure_blob/data/MatchTime")
)
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
                continue
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
            for p in plist:
                out.append({**p, "gate_label": 1})
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
        f_start = min(f_start, max(0, f_end - self.num_frames))
        if f_end - f_start < self.num_frames:
            f_end = min(len(vr) - 1, f_start + self.num_frames)
        idx = np.linspace(f_start, f_end, self.num_frames).astype(int)
        idx = np.clip(idx, 0, len(vr) - 1)
        frames = vr.get_batch(idx).asnumpy()
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
        labels = full_ids.clone()
        n = prompt_ids.size(-1)
        labels[..., :n] = -100
        return labels

    # ------------------------------------------------------------------
    def __getitem__(self, i):
        # Some clips trigger cv-preinfer "no canvases produced" (too-short
        # keyframe seek) or decord decode errors. Skip the offending sample
        # and try the next one; the entire training shouldn't die over a
        # handful of bad clips.
        n = len(self.samples)
        for attempt in range(n):
            idx = (i + attempt) % n
            try:
                return self._get_one(idx)
            except Exception as e:
                if attempt == 0:
                    print(f"[dataset] skip idx={idx}: {type(e).__name__}: {e}", flush=True)
        raise RuntimeError("no usable sample after full scan")

    def _get_one(self, i):
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
            text_prompt = text_full

        if self.video_backend == "frames":
            frames = self._load_frames_decord(s["video"], s["t"])
            frame_list = [frames[i] for i in range(frames.shape[0])]
            video_kwargs = {"videos": [frame_list], "num_frames": self.num_frames}
        else:
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
            labels = torch.full_like(full_ids, -100)

        item = {
            "input_ids": full_ids,
            "attention_mask": enc_full["attention_mask"][0],
            "labels": labels,
        }
        if self.stage == 2:
            item["gate_label"] = torch.tensor(s["gate_label"], dtype=torch.long)
            item["gate_pos"] = torch.tensor(full_ids.size(0) - 1, dtype=torch.long)

        for k in ("pixel_values", "image_grid_thw", "patch_positions",
                  "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"):
            if k in enc_full:
                item[k] = enc_full[k]
        return item


@dataclass
class OneVisionCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        assert len(batch) == 1, "Stage1/2 smoke uses per_device_batch_size=1"
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
                out[k] = item[k]
        return out
