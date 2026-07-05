"""Shared manifest -> segment expansion for the soccer streaming cache.

Both ``cut_clips.py`` and ``gen_codec.py`` import from here so the segment
list they iterate over is identical to what ``LazySupervisedDataset`` produces
at training time.
"""
from __future__ import annotations

import os
from typing import Iterable

import pandas as pd


def segments_for_row(row, cur_fps: int) -> list[tuple[str, float, float]]:
    """Replicate the trainer's per-row segment expansion.

    Each manifest row is one (game, half). We sort the row's annotated events
    in reverse temporal order (matching the training data loader) and emit
    ``(video_path, start, end)`` per event, where ``start`` is the previous
    event's timestamp (or 0 for the first event).
    """
    half = int(row["half"])
    anns = list(row["annotations"])
    ts_list: list[int] = []
    for ann in anns:
        gt = ann.get("gameTime", "")
        if " - " not in gt:
            continue
        head, mmss = gt.split(" - ", 1)
        try:
            h = int(head.strip().split(" ")[0])
        except Exception:
            continue
        if h != half:
            continue
        try:
            mm, ss = mmss.strip().split(":")
            ts = int(mm) * 60 + int(ss)
        except Exception:
            continue
        cap = ann.get("anonymized", ann.get("description", ""))
        if not cap:
            continue
        ts_list.append(ts)
    ts_list = ts_list[::-1]
    out: list[tuple[str, float, float]] = []
    vp = row["video_path"]
    for tid, ts in enumerate(ts_list):
        if tid == 0:
            start = min(0, ts - 1 / cur_fps)
            if start < 0:
                continue
        else:
            start = ts_list[tid - 1]
            if start == ts:
                continue
        out.append((vp, float(start), float(ts)))
    return out


def clip_path_for(video_path: str, start: float, end: float,
                  matchtime_root: str, clip_dir: str) -> tuple[str, float]:
    """Compute the trainer-compatible subclip path and duration."""
    duration = max(0.0, end - start)
    if duration <= 0:
        return "", 0.0
    game_key = os.path.relpath(video_path, matchtime_root).replace("/", "__")
    return (
        os.path.join(
            clip_dir,
            f"{game_key}__t{int(start * 100):08d}__d{int(duration * 100):08d}.mp4",
        ),
        duration,
    )


def iter_segments(manifest_path: str, matchtime_root: str, clip_dir: str,
                  cur_fps: int, start: int, limit: int | None
                  ) -> list[tuple[str, str, float, float]]:
    """Materialize the segment list for one manifest slice."""
    df = pd.read_parquet(manifest_path)
    n = len(df)
    end = min(n, start + limit) if limit is not None else n
    start = min(n, max(0, start))
    out: list[tuple[str, str, float, float]] = []
    for vid in range(start, end):
        row = df.iloc[vid]
        for vp, s, e in segments_for_row(row, cur_fps):
            clip, duration = clip_path_for(vp, s, e, matchtime_root, clip_dir)
            if not clip:
                continue
            out.append((vp, clip, s, duration))
    return out


def env_default(name: str, default: str | None = None) -> str | None:
    """Convenience: env var with optional default for argparse fallbacks."""
    val = os.environ.get(name)
    return val if val else default
