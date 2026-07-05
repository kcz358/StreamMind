"""Cut per-event subclips from raw SoccerNet/MatchTime videos.

Iterates the same manifest the trainer uses, derives ``(video_path, start, end)``
segments, and writes ``ffmpeg -c copy`` subclips into ``--clip_dir``.

Directories
-----------
--manifest        Path to ``train.parquet`` / ``valid.parquet``.
--matchtime_root  Dataset root; clip filenames are built from
                  ``relpath(video_path, matchtime_root)`` and MUST match the
                  trainer's ``MATCHTIME_ROOT`` exactly.
--clip_dir        Output directory for the cut subclips. The trainer also
                  reads from this path (typically a blob mount); the codec
                  cache key depends on the resulting file path.

Behaviour
---------
* Idempotent: existing non-empty clip files are skipped.
* No re-encode: ``-c copy`` keeps the original codec, so cuts snap to the
  nearest keyframe (~1s tolerance, acceptable for MatchTime supervision).
* Process pool: pass ``--workers`` to parallelise; each worker is a single
  ffmpeg invocation.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

from _manifest import env_default, iter_segments


def extract_subclip(src_video: str, t_start: float, duration: float,
                    out_path: str, timeout: int) -> str:
    """Atomic ``ffmpeg -c copy`` cut. Returns a short status string."""
    out = Path(out_path)
    if out.exists() and out.stat().st_size > 0:
        return "exists"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=out.suffix, prefix=".tmp_", dir=out.parent)
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{max(0.0, t_start):.3f}",
        "-t", f"{duration:.3f}",
        "-i", src_video,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=timeout, capture_output=True)
    except subprocess.CalledProcessError as exc:
        tmp_path.unlink(missing_ok=True)
        msg = exc.stderr.decode(errors="ignore")[:200] if exc.stderr else ""
        return f"ffmpeg_failed:{msg}"
    except subprocess.TimeoutExpired:
        tmp_path.unlink(missing_ok=True)
        return "ffmpeg_timeout"
    shutil.move(str(tmp_path), str(out_path))
    return "wrote"


def _worker(job: tuple[str, str, float, float, int]) -> tuple[str, str]:
    video_path, clip_path, start, duration, timeout = job
    if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
        return ("skip_exists", os.path.basename(clip_path))
    if not os.path.exists(video_path):
        return ("skip_novideo", os.path.basename(video_path))
    status = extract_subclip(video_path, start, duration, clip_path, timeout)
    if status == "exists":
        return ("skip_exists", os.path.basename(clip_path))
    if status == "wrote":
        return ("ok", os.path.basename(clip_path))
    return ("err", f"{status} :: {os.path.basename(clip_path)}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cut per-event subclips for the soccer streaming cache.")
    p.add_argument(
        "--manifest", required=True,
        help="Path to the *.parquet manifest used by the trainer.",
    )
    p.add_argument(
        "--matchtime_root",
        default=env_default("MATCHTIME_ROOT"),
        help="Dataset root used to derive game_key; MUST match the trainer's MATCHTIME_ROOT.",
    )
    p.add_argument(
        "--clip_dir",
        default=env_default("STREAMMIND_CLIP_CACHE"),
        help="Output directory for cut subclips (typically a blob mount).",
    )
    p.add_argument("--workers", type=int, default=int(env_default("CLIP_CUT_WORKERS", "8")))
    p.add_argument("--cur_fps", type=int, default=2)
    p.add_argument("--start", type=int, default=0, help="Manifest row offset.")
    p.add_argument("--limit", type=int, default=None, help="Manifest row count (default: all).")
    p.add_argument("--ffmpeg_timeout", type=int, default=180)
    p.add_argument("--log_every", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    for required in ("matchtime_root", "clip_dir"):
        if not getattr(args, required):
            raise SystemExit(f"--{required} is required (or set the equivalent environment variable).")
    os.makedirs(args.clip_dir, exist_ok=True)

    print(f"[cfg] manifest      = {args.manifest}", flush=True)
    print(f"[cfg] matchtime_root= {args.matchtime_root}", flush=True)
    print(f"[cfg] clip_dir      = {args.clip_dir}", flush=True)
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
        print("[manifest] nothing to do", flush=True)
        return

    jobs = [(vp, clip, s, dur, args.ffmpeg_timeout) for (vp, clip, s, dur) in segs]
    counts = {"ok": 0, "skip_exists": 0, "skip_novideo": 0, "err": 0}
    t0 = time.time()
    with Pool(args.workers, maxtasksperchild=200) as pool:
        for i, (status, info) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=4), 1):
            counts[status] = counts.get(status, 0) + 1
            if status == "err" and counts[status] <= 30:
                print(f"[err] {info}", flush=True)
            if i % args.log_every == 0 or i == len(jobs):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(jobs) - i) / max(rate, 1e-6)
                summary = " ".join(f"{k}={v}" for k, v in counts.items())
                print(
                    f"[{i}/{len(jobs)}] {summary} "
                    f"rate={rate:.1f}/s elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min",
                    flush=True,
                )

    print(
        f"[done] {' '.join(f'{k}={v}' for k, v in counts.items())} "
        f"elapsed={(time.time()-t0)/60:.1f}min",
        flush=True,
    )


if __name__ == "__main__":
    main()
