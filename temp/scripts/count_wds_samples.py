"""Count wds samples with >=1 <captioning>At turn across all shard indexes."""
from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def _count_shard(path: str) -> tuple[int, int]:
    total = 0
    kept = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            if "<captioning>At" in line:
                kept += 1
    return total, kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="/local_nvme/wds/index")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.index_dir, "*.jsonl")))
    print(f"[scan] {len(paths)} shard indexes  workers={args.workers}", flush=True)

    total = 0
    kept = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_count_shard, p): p for p in paths}
        done = 0
        for fut in as_completed(futs):
            t, k = fut.result()
            total += t
            kept += k
            done += 1
            print(
                f"[scan] {done}/{len(paths)}  total={total}  kept={kept}  "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
    print(f"\nTOTAL raw={total}  with_captioning={kept}", flush=True)


if __name__ == "__main__":
    main()
