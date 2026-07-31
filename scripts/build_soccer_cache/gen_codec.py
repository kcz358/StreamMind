"""Generate the per-segment codec cache used at training/eval time.

Iterates the same manifest the trainer uses, looks up each segment's clip
under ``--clip_dir`` (assumed already cut by ``cut_clips.py``), and runs the
OneVision-2 codec processor on it to write the canvas images + patch
positions under ``--codec_dir``.

Directories
-----------
--manifest        Path to ``train.parquet`` / ``valid.parquet``.
--matchtime_root  Dataset root used as the relpath base when deriving the
                  clip filename; MUST match the trainer's ``MATCHTIME_ROOT``.
--clip_dir        Directory of cut MP4 subclips (output of ``cut_clips.py``).
                  The cache key is derived from the resulting clip path, so
                  this MUST match the trainer's ``STREAMMIND_CLIP_CACHE``.
--codec_dir       Output directory for codec canvases / patch positions.
                  Becomes the trainer's ``ONLINE_CODEC_CACHE_DIR``.
--codec_module    Directory containing
                  ``codec_video_processing_llava_onevision2.py``; the
                  OneVision-2 model checkpoint dir works.

Behaviour
---------
* Skips segments whose ``meta.json`` + ``src_patch_position.npy`` already
  exist (idempotent).
* Skips segments whose clip is missing on disk (use ``cut_clips.py`` first).
* ``--patch`` and ``--max_pixels`` MUST match what the trainer's cache-hit
  precheck expects; the default values match the p16 streaming pipeline.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

from _manifest import env_default, iter_segments


_CODEC_CACHE_DIR_FOR = None
_RUN_CV_PREINFER = None
_LOAD_CODEC_RESULT = None
_MAYBE_WARN_SHORT_VIDEO = None
_CODEC_CFG = None
_CANONICAL_CLIP_DIR: str | None = None
_LOCAL_CLIP_DIR: str | None = None
import fcntl as _fcntl  # noqa: E402


def _canonical_to_local(canonical: str) -> str:
    if _LOCAL_CLIP_DIR is None or _CANONICAL_CLIP_DIR is None:
        return canonical
    rel = os.path.relpath(canonical, _CANONICAL_CLIP_DIR)
    return os.path.join(_LOCAL_CLIP_DIR, rel)


def _init_worker(codec_module_dir: str, codec_root: str,
                 patch: int, max_pixels: int,
                 canonical_clip_dir: str,
                 local_clip_dir: str | None) -> None:
    """Process-pool initializer; loads the codec processor once per worker."""
    global _CODEC_CACHE_DIR_FOR, _RUN_CV_PREINFER, _LOAD_CODEC_RESULT
    global _MAYBE_WARN_SHORT_VIDEO, _CODEC_CFG
    global _CANONICAL_CLIP_DIR, _LOCAL_CLIP_DIR
    os.environ["ONLINE_CODEC_CACHE_DIR"] = codec_root
    sys.path.insert(0, codec_module_dir)
    from codec_video_processing_llava_onevision2 import (
        CodecConfig, _cache_dir_for, _run_cv_preinfer,
        _load_codec_result, _maybe_warn_short_video,
    )
    _CODEC_CACHE_DIR_FOR = _cache_dir_for
    _RUN_CV_PREINFER = _run_cv_preinfer
    _LOAD_CODEC_RESULT = _load_codec_result
    _MAYBE_WARN_SHORT_VIDEO = _maybe_warn_short_video
    _CODEC_CFG = CodecConfig(patch=patch, max_pixels=max_pixels)
    _CANONICAL_CLIP_DIR = canonical_clip_dir
    _LOCAL_CLIP_DIR = local_clip_dir


def _worker(job: tuple[str, str, float, float]) -> tuple[str, str]:
    """Cache key uses the canonical clip path; IO reads from the local mirror."""
    _video_path, clip_path, _start, _duration = job
    try:
        out_dir = _CODEC_CACHE_DIR_FOR(clip_path, _CODEC_CFG)
        meta = out_dir / "meta.json"
        positions = out_dir / "src_patch_position.npy"
        if meta.exists() and positions.exists():
            return ("skip_exists", out_dir.name)
    except Exception as exc:
        return ("err", f"cachedir:{type(exc).__name__}:{str(exc)[:120]}")

    read_path = _canonical_to_local(clip_path)
    if not os.path.exists(read_path):
        return ("skip_noclip", os.path.basename(read_path))

    _MAYBE_WARN_SHORT_VIDEO(read_path, _CODEC_CFG)

    _CODEC_CFG.cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = _CODEC_CFG.cache_root / f".{out_dir.name}.lock"
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        _fcntl.flock(lock_fd, _fcntl.LOCK_EX)
        if meta.exists() and positions.exists():
            return ("skip_exists", out_dir.name)
        try:
            _RUN_CV_PREINFER(read_path, out_dir, _CODEC_CFG)
            return ("ok", out_dir.name)
        except Exception as exc:
            return ("err", f"{type(exc).__name__}:{str(exc)[:140]}")
    finally:
        try:
            _fcntl.flock(lock_fd, _fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the per-segment codec cache for the soccer streaming pipeline.",
    )
    p.add_argument("--manifest", required=True, help="Path to the *.parquet manifest used by the trainer.")
    p.add_argument(
        "--matchtime_root",
        default=env_default("MATCHTIME_ROOT"),
        help="Dataset root used to derive game_key; MUST match the trainer's MATCHTIME_ROOT.",
    )
    p.add_argument(
        "--clip_dir",
        default=env_default("STREAMMIND_CLIP_CACHE"),
        help="Canonical clip dir (drives the codec cache key; MUST match trainer's STREAMMIND_CLIP_CACHE).",
    )
    p.add_argument(
        "--local_clip_dir",
        default=None,
        help="Alternate dir where the mp4 files actually live (mirrors clip_dir layout); the codec key still uses clip_dir.",
    )
    p.add_argument(
        "--codec_dir",
        default=env_default("ONLINE_CODEC_CACHE_DIR"),
        help="Output directory for codec canvases / patch positions (becomes ONLINE_CODEC_CACHE_DIR).",
    )
    p.add_argument(
        "--codec_module",
        default=env_default("CODEC_MODULE_DIR"),
        help="Directory containing codec_video_processing_llava_onevision2.py.",
    )
    p.add_argument("--workers", type=int, default=int(env_default("CODEC_GEN_WORKERS", "8")))
    p.add_argument("--cur_fps", type=int, default=2)
    p.add_argument("--patch", type=int, default=16,
                   help="Codec patch size; MUST match the trainer's md5 key.")
    p.add_argument("--max_pixels", type=int, default=150_000,
                   help="Codec max_pixels; MUST match the trainer's md5 key.")
    p.add_argument("--start", type=int, default=0, help="Manifest row offset.")
    p.add_argument("--limit", type=int, default=None, help="Manifest row count (default: all).")
    p.add_argument("--log_every", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    for required in ("matchtime_root", "clip_dir", "codec_dir", "codec_module"):
        if not getattr(args, required):
            raise SystemExit(f"--{required} is required (or set the equivalent environment variable).")
    os.makedirs(args.codec_dir, exist_ok=True)

    print(f"[cfg] manifest      = {args.manifest}", flush=True)
    print(f"[cfg] matchtime_root= {args.matchtime_root}", flush=True)
    print(f"[cfg] clip_dir      = {args.clip_dir}", flush=True)
    print(f"[cfg] codec_dir     = {args.codec_dir}", flush=True)
    print(f"[cfg] codec_module  = {args.codec_module}", flush=True)
    print(f"[cfg] patch={args.patch}  max_pixels={args.max_pixels}  workers={args.workers}", flush=True)

    t0 = time.time()
    jobs = iter_segments(
        manifest_path=args.manifest,
        matchtime_root=args.matchtime_root,
        clip_dir=args.clip_dir,
        cur_fps=args.cur_fps,
        start=args.start,
        limit=args.limit,
    )
    print(f"[manifest] segments={len(jobs)} build_time={time.time()-t0:.1f}s", flush=True)
    if not jobs:
        print("[manifest] nothing to do", flush=True)
        return

    counts = {"ok": 0, "skip_exists": 0, "skip_noclip": 0, "err": 0}
    t0 = time.time()
    with Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(args.codec_module, args.codec_dir, args.patch, args.max_pixels,
                  args.clip_dir, args.local_clip_dir),
        maxtasksperchild=200,
    ) as pool:
        for i, (status, info) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=2), 1):
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
