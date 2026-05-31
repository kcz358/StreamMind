# ov_train/codec_utils.py
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
    """Cut [t_start, t_start+duration] from src_video -> out_path. Idempotent.

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

    shutil.move(str(tmp_path), str(out_path))
    return out_path
