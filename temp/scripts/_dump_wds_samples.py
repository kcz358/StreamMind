import argparse
import json
import os
import tarfile

WANTED_TYPES = ("<captioning>", "<memory>")


def choose_sample_ids(manifest_path, n, need_captioning=True, need_memory=False):
    picked = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = row.get("messages", [])
            texts = " ".join(m.get("content", "") for m in msgs if isinstance(m, dict))
            ok_cap = ("<captioning>At" in texts) if need_captioning else True
            ok_mem = ("<memory>" in texts) if need_memory else True
            if ok_cap and ok_mem:
                picked.append(row)
                if len(picked) >= n:
                    break
    return picked


def dump_samples(tar_path, samples, out_root):
    id_to_frames = {}
    for s in samples:
        id_to_frames[s["id"]] = {name: None for name in s.get("images", [])}
    sample_dirs = {}
    for s in samples:
        d = os.path.join(out_root, s["id"])
        os.makedirs(d, exist_ok=True)
        sample_dirs[s["id"]] = d
        with open(os.path.join(d, "manifest_row.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)

    all_needed = set()
    for s in samples:
        all_needed.update(s.get("images", []))

    remaining = set(all_needed)
    with tarfile.open(tar_path, mode="r|") as tf:
        for m in tf:
            if m.name in remaining:
                sid = m.name.split("/", 2)[1]
                out_dir = sample_dirs[sid]
                os.makedirs(out_dir, exist_ok=True)
                f = tf.extractfile(m)
                if f is None:
                    continue
                with open(os.path.join(out_dir, os.path.basename(m.name)), "wb") as g:
                    g.write(f.read())
                remaining.remove(m.name)
                if not remaining:
                    break
    return sorted(all_needed - remaining), sorted(remaining)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_samples", type=int, default=3)
    ap.add_argument("--require_memory", action="store_true")
    args = ap.parse_args()

    samples = choose_sample_ids(args.manifest, args.num_samples, need_memory=args.require_memory)
    if not samples:
        raise SystemExit("no samples matched criteria")
    print("picked ids:", [s["id"] for s in samples])
    got, missing = dump_samples(args.tar, samples, args.out)
    print(f"wrote {len(got)} frames to {args.out}")
    if missing:
        print("MISSING:", missing[:5], "..." if len(missing) > 5 else "")
