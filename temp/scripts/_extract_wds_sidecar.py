import argparse
import os
import sys
import tarfile


def extract(tar_path, targets, out_dir):
    remaining = set(targets)
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tar_path, mode="r|") as tf:
        for m in tf:
            if m.name in remaining:
                f = tf.extractfile(m)
                if f is None:
                    continue
                out_path = os.path.join(out_dir, os.path.basename(m.name))
                with open(out_path, "wb") as g:
                    g.write(f.read())
                print(f"wrote {out_path} ({m.size} bytes)")
                remaining.remove(m.name)
                if not remaining:
                    return
    if remaining:
        print("MISSING:", remaining, file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tar")
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    args = ap.parse_args()
    extract(args.tar, args.names, args.out)
