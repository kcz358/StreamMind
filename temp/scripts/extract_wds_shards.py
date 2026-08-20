import argparse
import json
import os
import sys
import tarfile
import time


def process_shard(tar_path, ext_root, index_dir, dry_run=False):
    part = os.path.basename(os.path.dirname(tar_path))
    shard = os.path.basename(tar_path).removesuffix(".tar")
    out_dir = os.path.join(ext_root, part, shard)
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)

    manifest_lines = []
    n_files = 0
    n_bytes = 0
    t0 = time.time()
    with tarfile.open(tar_path, mode="r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            data = f.read()
            n_files += 1
            n_bytes += len(data)
            out_path = os.path.join(out_dir, m.name)
            if dry_run:
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as g:
                g.write(data)
            if m.name.endswith("manifest.jsonl"):
                manifest_lines = data.splitlines()

    dt = time.time() - t0
    n_samples = 0
    if not dry_run:
        shard_index_path = os.path.join(index_dir, f"{part}__{shard}.jsonl")
        with open(shard_index_path, "w", encoding="utf-8") as g:
            for line in manifest_lines:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = row["id"]
                row["shard"] = f"{part}/{shard}"
                row["shard_root"] = out_dir
                row["images_abs"] = [os.path.join(out_dir, p) for p in row.get("images", [])]
                g.write(json.dumps(row) + "\n")
                n_samples += 1
    print(
        f"[done] {part}/{shard}\tfiles={n_files}\tbytes={n_bytes/1e9:.1f}G\tsamples={n_samples}\ttime={dt:.1f}s",
        flush=True,
    )
    return n_files, n_bytes, n_samples, dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", nargs="+", required=True)
    ap.add_argument("--ext_root", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    for t in args.tars:
        process_shard(t, args.ext_root, args.index_dir, dry_run=args.dry_run)
