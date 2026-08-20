import argparse
import os
import sys
import tarfile
import time


def peek(tar_path, max_head=5):
    t0 = time.time()
    manifest_offset = None
    first_names = []
    with tarfile.open(tar_path, mode="r|") as tf:
        for i, m in enumerate(tf):
            if len(first_names) < max_head:
                first_names.append(m.name)
            if m.name == "manifest.jsonl":
                manifest_offset = i
                break
    dt = time.time() - t0
    print(f"{os.path.basename(tar_path)}\tmanifest_at_member={manifest_offset}\ttime={dt:.2f}s\tfirst={first_names}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tars", nargs="+")
    args = ap.parse_args()
    for t in args.tars:
        peek(t)
