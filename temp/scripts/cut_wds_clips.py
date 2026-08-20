"""Cut per-caption wds subclips directly from decoded jpg frames.

Instead of first stitching each 180s sample back to a full mp4 and then
running ``ffmpeg -c copy`` on it, this script picks the frames whose
timestamps fall inside ``[start, end]`` for a given segment and encodes
them straight to the final clip mp4.

Directories mirror the soccer pipeline:

* ``--manifest`` : ``manifests/wds_180s.parquet`` from
  ``build_wds_manifest.py``.
* ``--sample_map``: sibling ``*.sample_map.jsonl`` mapping each
  ``sample_id`` to its decoded frame directory.
* ``--matchtime_root`` / ``--clip_dir``: same env vars the trainer uses;
  clip filenames are derived by ``_manifest.clip_path_for`` so the codec
  cache key matches exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _manifest import env_default, iter_segments  # noqa: E402


def _load_sample_map(path: str, keep_paths: set[str] | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if keep_paths is not None and row["video_path"] not in keep_paths:
                continue
            out[row["video_path"]] = row
    return out


def _frame_time(name: str, fps: float) -> float:
    stem = os.path.splitext(os.path.basename(name))[0]
    idx = int(stem.split("_")[-1])
    return idx / max(fps, 1e-6)


def _select_frames(frames_abs: list[str], fps: float, start: float, end: float) -> list[str]:
    picked = []
    for p in frames_abs:
        t = _frame_time(p, fps)
        if start - 1e-6 <= t <= end + 1e-6:
            picked.append((t, p))
    picked.sort(key=lambda x: x[0])
    return [p for _t, p in picked]


def _encode_clip(frames: list[str], out_path: str, timeout: int) -> str:
    out = Path(out_path)
    if out.exists() and out.stat().st_size > 0:
        return "exists"
    if not frames:
        return "no_frames"
    out.parent.mkdir(parents=True, exist_ok=True)

    list_fd, list_name = tempfile.mkstemp(suffix=".txt", prefix=".wds_", dir=out.parent)
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=out.suffix, prefix=".tmp_", dir=out.parent)
    os.close(list_fd)
    os.close(tmp_fd)
    try:
        with open(list_name, "w", encoding="utf-8") as f:
            for p in frames:
                f.write(f"file '{p}'\n")
                f.write("duration 1\n")
            f.write(f"file '{frames[-1]}'\n")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-r", "1",
            "-i", list_name,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-r", "1",
            tmp_name,
        ]
        try:
            subprocess.run(cmd, check=True, timeout=timeout, capture_output=True)
        except subprocess.CalledProcessError as exc:
            msg = exc.stderr.decode(errors="ignore")[:200] if exc.stderr else ""
            return f"ffmpeg_failed:{msg}"
        except subprocess.TimeoutExpired:
            return "ffmpeg_timeout"
        shutil.move(tmp_name, out_path)
        return "wrote"
    finally:
        Path(list_name).unlink(missing_ok=True)
        Path(tmp_name).unlink(missing_ok=True)


_SAMPLE_MAP: dict[str, dict] = {}
_FFMPEG_TIMEOUT = 180


def _init_worker(
    sample_map_path: str,
    keep_paths: set[str] | None,
    timeout: int,
    canonical_clip_dir: str,
    local_clip_dir: str | None,
) -> None:
    global _SAMPLE_MAP, _FFMPEG_TIMEOUT, _CANONICAL_CLIP_DIR, _LOCAL_CLIP_DIR
    _SAMPLE_MAP = _load_sample_map(sample_map_path, keep_paths=keep_paths)
    _FFMPEG_TIMEOUT = timeout
    _CANONICAL_CLIP_DIR = canonical_clip_dir
    _LOCAL_CLIP_DIR = local_clip_dir


def _worker(job: tuple[str, str, float, float]) -> tuple[str, str]:
    video_path, clip_path, start, duration = job
    write_path = _canonical_to_local(clip_path)
    if os.path.exists(write_path) and os.path.getsize(write_path) > 0:
        return ("skip_exists", os.path.basename(write_path))
    info = _SAMPLE_MAP.get(video_path)
    if info is None:
        return ("skip_nomap", os.path.basename(write_path))
    fps = float(info.get("fps") or 1.0)
    frames = _select_frames(info["images_abs"], fps, start, start + duration)
    if not frames:
        return ("skip_noframes", os.path.basename(write_path))
    status = _encode_clip(frames, write_path, _FFMPEG_TIMEOUT)
    if status == "wrote":
        return ("ok", os.path.basename(write_path))
    if status == "exists":
        return ("skip_exists", os.path.basename(write_path))
    return ("err", f"{status} :: {os.path.basename(write_path)}")


_LOCAL_CLIP_DIR: str | None = None
_CANONICAL_CLIP_DIR: str | None = None


def _canonical_to_local(canonical: str) -> str:
    if _LOCAL_CLIP_DIR is None or _CANONICAL_CLIP_DIR is None:
        return canonical
    rel = os.path.relpath(canonical, _CANONICAL_CLIP_DIR)
    return os.path.join(_LOCAL_CLIP_DIR, rel)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--sample_map", required=True)
    p.add_argument(
        "--matchtime_root",
        default=env_default("MATCHTIME_ROOT"),
    )
    p.add_argument(
        "--clip_dir",
        default=env_default("STREAMMIND_CLIP_CACHE"),
        help="Canonical clip dir (used to build clip names / codec cache key).",
    )
    p.add_argument(
        "--local_clip_dir",
        default=None,
        help="If set, write mp4 files here (mirroring clip_dir layout) instead of clip_dir directly.",
    )
    p.add_argument("--workers", type=int, default=int(env_default("CLIP_CUT_WORKERS", "8")))
    p.add_argument("--cur_fps", type=int, default=2)
    p.add_argument("--start", type=int, default=0, help="Manifest row offset.")
    p.add_argument("--limit", type=int, default=None, help="Manifest row count (default: all).")
    p.add_argument("--ffmpeg_timeout", type=int, default=180)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    for required in ("matchtime_root", "clip_dir"):
        if not getattr(args, required):
            raise SystemExit(f"--{required} required")
    if args.local_clip_dir:
        os.makedirs(args.local_clip_dir, exist_ok=True)

    print(f"[cfg] manifest      = {args.manifest}", flush=True)
    print(f"[cfg] sample_map    = {args.sample_map}", flush=True)
    print(f"[cfg] matchtime_root= {args.matchtime_root}", flush=True)
    print(f"[cfg] clip_dir(key) = {args.clip_dir}", flush=True)
    print(f"[cfg] local_clip_dir= {args.local_clip_dir or '<same as clip_dir>'}", flush=True)
    print(f"[cfg] workers       = {args.workers}", flush=True)

    t0 = time.time()
    segs = iter_segments(
        manifest_path=args.manifest,
        matchtime_root=args.matchtime_root,
        clip_dir=args.clip_dir,
        cur_fps=args.cur_fps,
        start=args.start,
        limit=args.limit,
    )
    print(f"[manifest] segments={len(segs)} build_time={time.time()-t0:.1f}s", flush=True)
    if not segs:
        return

    counts: dict[str, int] = {}
    t0 = time.time()
    keep_paths = {vp for vp, _, _, _ in segs}
    with Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(args.sample_map, keep_paths, args.ffmpeg_timeout, args.clip_dir, args.local_clip_dir),
        maxtasksperchild=200,
    ) as pool:
        for i, (status, info) in enumerate(pool.imap_unordered(_worker, segs, chunksize=4), 1):
            counts[status] = counts.get(status, 0) + 1
            if status == "err" and counts[status] <= 30:
                print(f"[err] {info}", flush=True)
            if i % args.log_every == 0 or i == len(segs):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(segs) - i) / max(rate, 1e-6)
                summary = " ".join(f"{k}={v}" for k, v in counts.items())
                print(
                    f"[{i}/{len(segs)}] {summary} "
                    f"rate={rate:.1f}/s elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min",
                    flush=True,
                )

    print(f"[done] {' '.join(f'{k}={v}' for k, v in counts.items())}", flush=True)


if __name__ == "__main__":
    main()
