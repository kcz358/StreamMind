import argparse
import json
import os
import tarfile
from collections import Counter


def scan(path):
    exts = Counter()
    top_dirs = Counter()
    n = 0
    with tarfile.open(path, mode="r|") as tf:
        for m in tf:
            n += 1
            name = m.name
            base = os.path.basename(name)
            _, ext = os.path.splitext(base)
            exts[ext.lower() or "<noext>"] += 1
            parts = name.split("/", 1)
            top_dirs[parts[0]] += 1
    print(json.dumps({
        "path": path,
        "size_bytes": os.path.getsize(path),
        "total_members": n,
        "ext_counts": dict(exts.most_common()),
        "top_dir_counts": dict(top_dirs.most_common()),
    }, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tar")
    args = ap.parse_args()
    scan(args.tar)
