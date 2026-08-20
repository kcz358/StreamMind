import argparse
import os
import shutil
import sys
import time


def find_incomplete(ext_root, index_dir):
    done = set()
    for fn in os.listdir(index_dir):
        if fn.endswith(".jsonl"):
            done.add(fn[:-6])
    incomplete = []
    for part in sorted(os.listdir(ext_root)):
        part_dir = os.path.join(ext_root, part)
        if not os.path.isdir(part_dir):
            continue
        for shard in sorted(os.listdir(part_dir)):
            shard_dir = os.path.join(part_dir, shard)
            if not os.path.isdir(shard_dir):
                continue
            key = f"{part}__{shard}"
            if key in done:
                continue
            incomplete.append(shard_dir)
    return incomplete


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext_root", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--yes", action="store_true", help="actually delete")
    args = ap.parse_args()

    to_del = find_incomplete(args.ext_root, args.index_dir)
    print(f"[info] {len(to_del)} incomplete shards to remove:", flush=True)
    for p in to_del:
        print(f"  - {p}", flush=True)
    if not args.yes:
        print("[dry-run] pass --yes to actually delete", flush=True)
        sys.exit(0)

    total_start = time.time()
    for i, p in enumerate(to_del, 1):
        t0 = time.time()
        shutil.rmtree(p, ignore_errors=False)
        print(f"[{i}/{len(to_del)}] removed {p} ({time.time()-t0:.1f}s)", flush=True)
    print(f"[done] total {time.time()-total_start:.1f}s", flush=True)
