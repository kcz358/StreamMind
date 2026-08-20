"""Sample N wds records and emit a soccer-style parquet + sample map.

Reads the per-shard ``index/*.jsonl`` files produced by ``extract_wds_shards``
and turns them into a manifest identical in schema to the soccer parquet so
that ``_manifest.iter_segments`` + downstream ``gen_codec.py`` work with zero
modifications.

Shards are scanned in parallel across ``--workers`` processes. Each worker
parses one full shard, keeps only rows that have at least one
``<captioning>At X - Y seconds:`` turn, and returns a compact dict list.

Per row emitted to parquet:

    video_path : virtual path under matchtime_root, e.g.
                 ``wds_180s/<sample_id>.mp4`` (never actually decoded; only
                 used as a stable key for clip filename derivation).
    half       : always 1 (needed by ``segments_for_row``).
    annotations: list of ``{gameTime: "1 - MM:SS", anonymized: <caption>}``.

A companion ``.sample_map.jsonl`` records where each sample_id's decoded
frame directory lives so ``cut_wds_clips.py`` can find the jpg sequence.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


CAP_RE = re.compile(r"<captioning>At\s+(\d+)\s*-\s*(\d+)\s*seconds?:\s*(.+)", re.S)


def _extract_annotations(messages: list[dict]) -> list[dict]:
    anns = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        c = m.get("content", "")
        mobj = CAP_RE.match(c)
        if not mobj:
            continue
        ts = int(mobj.group(2))
        caption = mobj.group(3).strip()
        anns.append(
            {
                "gameTime": f"1 - {ts // 60:02d}:{ts % 60:02d}",
                "anonymized": caption,
                "description": caption,
            }
        )
    return anns


def _process_shard(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if "<captioning>At" not in line:
                continue
            row = json.loads(line)
            anns = _extract_annotations(row.get("messages", []))
            if not anns:
                continue
            out.append(
                {
                    "sample_id": row["id"],
                    "shard_root": row["shard_root"],
                    "images_abs": row["images_abs"],
                    "annotations": anns,
                    "fps": row.get("fps"),
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="/local_nvme/wds/index")
    ap.add_argument(
        "--matchtime_root",
        default=os.environ.get("MATCHTIME_ROOT", "/data/kaichen/data/MatchTime"),
        help="Trainer's MATCHTIME_ROOT; determines the virtual clip filename prefix.",
    )
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--out_sample_map", required=True)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--virtual_video_prefix",
        default="wds_180s",
        help="Subdir under matchtime_root used to build video_path (never opened).",
    )
    args = ap.parse_args()

    import pandas as pd

    paths = sorted(glob.glob(os.path.join(args.index_dir, "*.jsonl")))
    print(f"[scan] {len(paths)} shard indexes  workers={args.workers}", flush=True)
    results: dict[str, list[dict]] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_shard, p): p for p in paths}
        completed = 0
        for fut in as_completed(futs):
            path = futs[fut]
            rows = fut.result()
            results[path] = rows
            completed += 1
            dt = time.time() - t0
            print(
                f"[scan] {completed}/{len(paths)} done  +{len(rows)} rows  "
                f"elapsed={dt:.1f}s",
                flush=True,
            )
    pool: list[dict] = []
    for path in paths:
        pool.extend(results[path])

    print(f"[info] final pool size = {len(pool)}", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    picked = pool[: args.num_samples]

    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_sample_map), exist_ok=True)

    rows = []
    with open(args.out_sample_map, "w", encoding="utf-8") as f_map:
        for s in picked:
            sample_id = s["sample_id"]
            video_path = os.path.join(
                args.matchtime_root, args.virtual_video_prefix, f"{sample_id}.mp4"
            )
            rows.append(
                {
                    "video_path": video_path,
                    "caption_json_path": "",
                    "source": "wds",
                    "data_type": "wds_180s",
                    "half": 1,
                    "annotations": s["annotations"],
                    "num_captions": len(s["annotations"]),
                }
            )
            f_map.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "video_path": video_path,
                        "shard_root": s["shard_root"],
                        "images_abs": s["images_abs"],
                        "fps": s["fps"],
                    }
                )
                + "\n"
            )
    df = pd.DataFrame(rows)
    df.to_parquet(args.out_parquet, index=False)
    print(f"[wrote] parquet={args.out_parquet}  rows={len(df)}", flush=True)
    print(f"[wrote] sample_map={args.out_sample_map}", flush=True)


if __name__ == "__main__":
    main()
