"""Build a parquet manifest of all soccer-format games (SoccerNet + streaming_matchtime).

One row per game:
  video_path: str         - full path to .mkv or .mp4 (the source video for ffmpeg cut)
  caption_json_path: str  - full path to Labels-caption.json
  source: str             - "matchtime" | "inf-stream" | "live_cc" | "live_whisperx"
  data_type: str          - "train" | "valid"
  half: int               - 1 or 2 (SoccerNet); always 1 for streaming_matchtime
  annotations: list[dict] - {gameTime, anonymized} parsed from json
  num_captions: int

Run on the pod:
  python -m ov_train.build_manifest \\
    --matchtime_root /data/kaichen/data/MatchTime \\
    --streaming_root /data/project_gen/dataset_understand/streaming_matchtime \\
    --out_dir /data/kaichen/data/MatchTime/manifests
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matchtime_root", default="/data/kaichen/data/MatchTime")
    p.add_argument("--streaming_root", default="/data/project_gen/dataset_understand/streaming_matchtime")
    p.add_argument("--out_dir", default="/data/kaichen/data/MatchTime/manifests")
    return p.parse_args()


def _load_captions(json_path: str) -> Optional[list[dict]]:
    try:
        data = json.load(open(json_path))
    except Exception:
        return None
    out = []
    for ann in data.get("annotations", []):
        gt = ann.get("gameTime", "")
        text = ann.get("anonymized", ann.get("description", ""))
        if not gt or not text:
            continue
        out.append({"gameTime": gt, "anonymized": text})
    return out


def iter_matchtime(root: str, data_type: str) -> Iterator[dict]:
    """SoccerNet layout: features_video/<league>/<game>/{1,2}_224p.mkv with paired
    dataset/MatchTime/<data_type>/<league>/<game>/Labels-caption.json."""
    video_root = Path(root) / "features_video"
    if not video_root.exists():
        return
    for league_dir in sorted(video_root.iterdir()):
        if not league_dir.is_dir():
            continue
        for game_dir in sorted(league_dir.iterdir()):
            if not game_dir.is_dir():
                continue
            for half_name in ("1_224p.mkv", "2_224p.mkv"):
                video_path = game_dir / half_name
                if not video_path.exists():
                    continue
                json_path = Path(root) / "dataset" / "MatchTime" / data_type / league_dir.name / game_dir.name / "Labels-caption.json"
                if not json_path.exists():
                    continue
                anns = _load_captions(str(json_path))
                if not anns:
                    continue
                half = int(half_name.split("_")[0])
                # filter annotations belonging to this half (gameTime "<half> - MM:SS")
                half_anns = [a for a in anns if a["gameTime"].startswith(f"{half} -")]
                if not half_anns:
                    continue
                yield {
                    "video_path": str(video_path),
                    "caption_json_path": str(json_path),
                    "source": "matchtime",
                    "data_type": data_type,
                    "half": half,
                    "annotations": half_anns,
                    "num_captions": len(half_anns),
                }


def iter_streaming(root: str, data_type: str) -> Iterator[dict]:
    """streaming_matchtime: per-game dir under dataset/MatchTime/<data_type>/<src>/<game>/Labels-caption.json
    + pre-cut clip at clips/features_video__<src>__<game>__1_224p.mkv__t00000000__d<dur>.mp4."""
    ds_root = Path(root) / "dataset" / "MatchTime" / data_type
    clips_root = Path(root) / "clips"
    if not ds_root.exists() or not clips_root.exists():
        return

    # Pre-index clips by (src, game) so we don't rescan 543k files per game.
    print(f"[iter_streaming] indexing clips...")
    clip_index = {}
    for clip in clips_root.iterdir():
        name = clip.name
        if not name.startswith("features_video__"):
            continue
        rest = name[len("features_video__"):]
        if "__1_224p.mkv__" not in rest:
            continue
        head, _, _ = rest.partition("__1_224p.mkv__")
        src, _, game = head.partition("__")
        clip_index[(src, game)] = str(clip)
    print(f"[iter_streaming] indexed {len(clip_index)} clips")

    for src_dir in sorted(ds_root.iterdir()):
        if not src_dir.is_dir():
            continue
        src = src_dir.name
        for game_dir in sorted(src_dir.iterdir()):
            if not game_dir.is_dir():
                continue
            json_path = game_dir / "Labels-caption.json"
            if not json_path.exists():
                continue
            anns = _load_captions(str(json_path))
            if not anns:
                continue
            clip = clip_index.get((src, game_dir.name))
            if not clip:
                continue
            half = 1
            half_anns = [a for a in anns if a["gameTime"].startswith(f"{half} -")]
            if not half_anns:
                continue
            yield {
                "video_path": clip,
                "caption_json_path": str(json_path),
                "source": src,
                "data_type": data_type,
                "half": half,
                "annotations": half_anns,
                "num_captions": len(half_anns),
            }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for split in ("train", "valid"):
        rows = []
        if args.matchtime_root and os.path.isdir(args.matchtime_root):
            rows.extend(iter_matchtime(args.matchtime_root, split))
        if args.streaming_root and os.path.isdir(args.streaming_root):
            rows.extend(iter_streaming(args.streaming_root, split))
        print(f"[{split}] {len(rows)} games")
        if not rows:
            continue
        df = pd.DataFrame(rows)
        out_path = os.path.join(args.out_dir, f"{split}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"[{split}] wrote {out_path}")
        print(df.groupby("source").size())


if __name__ == "__main__":
    main()
