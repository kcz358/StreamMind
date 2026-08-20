"""Estimate training ETA from a torchrun rank stdout log.

Parses ``'epoch': '<float>'`` markers to infer the number of optimizer
steps completed (each marker == one training step under num_epochs=1
scheduling), then combines that with the total elapsed wall clock to
extrapolate a total-run duration and remaining time.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--total_rows", type=int, default=3345303,
                    help="Rows in the merged train parquet.")
    ap.add_argument("--per_row_segments", type=float, default=24.4,
                    help="Estimated segments per parquet row (13.3M/543k soccer baseline).")
    args = ap.parse_args()

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    epochs = [float(m.group(1)) for m in re.finditer(r"'epoch': '([0-9.eE+\-]+)'", data)]
    if not epochs:
        print("[eta] no 'epoch' markers found in log yet", file=sys.stderr)
        sys.exit(2)
    last_epoch = epochs[-1]
    n_updates = len(epochs)

    log_start = os.path.getmtime(args.log)
    log_start = min(log_start, time.time())
    log_mtime = os.path.getmtime(args.log)
    log_start_ctime = os.path.getctime(args.log)
    elapsed_seconds = log_mtime - log_start_ctime
    if elapsed_seconds <= 0:
        elapsed_seconds = 1.0

    frac_done = last_epoch
    if frac_done <= 0:
        print("[eta] last_epoch is 0, cannot estimate")
        sys.exit(3)
    total_seconds = elapsed_seconds / frac_done
    remaining_seconds = total_seconds - elapsed_seconds

    def fmt(sec: float) -> str:
        h = sec / 3600.0
        d = h / 24.0
        if d >= 1.5:
            return f"{d:.2f} d ({h:.1f} h)"
        return f"{h:.2f} h"

    print(f"[eta] log         = {args.log}")
    print(f"[eta] updates     = {n_updates}")
    print(f"[eta] last_epoch  = {last_epoch:.6f}")
    print(f"[eta] elapsed     = {fmt(elapsed_seconds)}")
    print(f"[eta] total_est   = {fmt(total_seconds)}")
    print(f"[eta] remaining   = {fmt(remaining_seconds)}")
    print(f"[eta] rate        = {n_updates/max(elapsed_seconds,1):.2f} updates/s")


if __name__ == "__main__":
    main()
