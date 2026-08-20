import argparse
import json
import os
import tarfile
from collections import Counter


def probe(path, max_members):
    size = os.path.getsize(path)
    exts = Counter()
    sample_names = []
    sample_bytes = {}
    stems = set()
    n = 0
    with tarfile.open(path, mode="r|") as tf:
        for m in tf:
            n += 1
            if not m.isfile():
                continue
            name = m.name
            base = os.path.basename(name)
            stem, ext = os.path.splitext(base)
            exts[ext.lower() or "<noext>"] += 1
            if len(sample_names) < 20:
                sample_names.append((name, m.size))
            if ext.lower() in {".json", ".txt", ".jsonl"} and stem not in sample_bytes:
                f = tf.extractfile(m)
                if f is not None:
                    sample_bytes[name] = f.read(4096)
            stems.add(stem.split(".")[0])
            if n >= max_members:
                break
    print(json.dumps({
        "path": path,
        "size_bytes": size,
        "members_scanned": n,
        "ext_counts": dict(exts.most_common()),
        "unique_stem_prefix": len(stems),
        "sample_names": sample_names,
    }, indent=2))
    for name, data in sample_bytes.items():
        print("---", name, "---")
        try:
            print(data.decode("utf-8", errors="replace"))
        except Exception as e:
            print("<binary>", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tar")
    ap.add_argument("--max_members", type=int, default=200)
    args = ap.parse_args()
    probe(args.tar, args.max_members)
