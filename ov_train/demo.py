"""Streaming demo for StreamMind × LLaVA-OneVision-2.

Plays one MatchTime half by sliding an 8-second window every `--step` seconds.
At every step:
  1. Encode the 8s clip with the codec backend (same as training).
  2. Gate forward -> [silence, response] probabilities.
  3. If response, call base.generate to produce one-line commentary.

Output: text timeline on stdout. Optional --output_video writes an MP4 with
generated captions burned in as subtitles.

Example:
  python -m ov_train.demo \\
    --resume_from /data/kaichen/StreamMind/stage2_codec_.../checkpoint-13982 \\
    --video      /data/kaichen/data/MatchTime/features_video/.../1_224p.mkv \\
    --captions   /data/kaichen/data/MatchTime/dataset/MatchTime/valid/.../Labels-caption.json \\
    --t_start 0 --t_end 600 --step 2 \\
    --output_video /tmp/demo.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ov_train.codec_utils import extract_subclip
from ov_train.onevision_stream import StreamOneVision
from ov_train.soccer_dataset import (
    CODEC_CONFIG,
    CODEC_MAX_PIXELS,
    SYSTEM_PROMPT,
    USER_PROMPT,
    _parse_game_time,
)


# ---------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/data/kaichen/LLaVA-OneVision-1.5-RL/pretrained/LLaVA-OneVision-2-8B-Instruct")
    p.add_argument("--resume_from", required=True, help="stage2 checkpoint dir")
    p.add_argument("--video", required=True, help="full game-half mkv")
    p.add_argument("--captions", default=None, help="optional Labels-caption.json for ground-truth overlay")
    p.add_argument("--half", type=int, default=1, help="which half (1 or 2) to filter captions by")
    p.add_argument("--t_start", type=float, default=0.0)
    p.add_argument("--t_end", type=float, default=300.0)
    p.add_argument("--step", type=float, default=2.0, help="sliding stride in seconds")
    p.add_argument("--window", type=float, default=8.0, help="context window in seconds")
    p.add_argument("--gate_thresh", type=float, default=0.5,
                   help="response prob >= this triggers generation")
    p.add_argument("--max_new_tokens", type=int, default=60)
    p.add_argument("--clip_cache", default="/tmp/demo_clips")
    p.add_argument("--codec_cache", default="/tmp/demo_codec_cache")
    p.add_argument("--output_video", default=None, help="if set, burn subtitles into mp4")
    return p.parse_args()


# ---------------------------------------------------------------- helpers
def build_prompt_messages():
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]


def load_gt_captions(path: str | None, half: int) -> list[tuple[float, str]]:
    if not path:
        return []
    data = json.loads(Path(path).read_text())
    out = []
    for ann in data.get("annotations", []):
        gt = ann.get("gameTime")
        text = ann.get("anonymized") or ann.get("description")
        if not gt or not text:
            continue
        try:
            h, t = _parse_game_time(gt)
        except Exception:
            continue
        if h == half:
            out.append((float(t), text.strip()))
    return sorted(out)


def secs_to_srt_ts(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = t - 3600 * h - 60 * m
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def write_srt(events: list[tuple[float, str]], path: str, t_start: float, dur_per_caption: float = 4.0):
    """Each caption shown for `dur_per_caption` seconds (or until next one)."""
    with open(path, "w") as f:
        for i, (t, text) in enumerate(events):
            t0 = max(0.0, t - t_start)
            t1 = t0 + dur_per_caption
            if i + 1 < len(events):
                t1 = min(t1, max(0.0, events[i + 1][0] - t_start))
            f.write(f"{i + 1}\n{secs_to_srt_ts(t0)} --> {secs_to_srt_ts(t1)}\n{text}\n\n")


def burn_subs(src_video: str, srt_path: str, t_start: float, t_end: float, out_path: str):
    dur = t_end - t_start
    # Escape srt path for ffmpeg subtitles filter (commas/colons in paths break it).
    safe_srt = srt_path.replace(":", r"\:").replace(",", r"\,")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t_start:.3f}", "-t", f"{dur:.3f}",
        "-i", src_video,
        "-vf", f"subtitles='{safe_srt}'",
        "-c:a", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    Path(args.clip_cache).mkdir(parents=True, exist_ok=True)
    os.environ["ONLINE_CODEC_CACHE_DIR"] = args.codec_cache
    Path(args.codec_cache).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] loading processor + model... (this takes ~1 min)", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    model = StreamOneVision(args.model_path)
    state = load_file(os.path.join(args.resume_from, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[demo] resumed: {len(state)} tensors loaded "
          f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    model = model.to(device).eval()

    gts = load_gt_captions(args.captions, args.half)
    print(f"[demo] ground-truth captions in half {args.half}: {len(gts)}", flush=True)

    prompt_msgs = build_prompt_messages()
    text_prompt = processor.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )

    events: list[tuple[float, str]] = []
    t = args.t_start + args.window
    while t <= args.t_end:
        clip = Path(args.clip_cache) / f"demo_t{int(t * 100):08d}.mp4"
        try:
            extract_subclip(args.video, t - args.window, args.window, clip)
        except Exception as e:
            print(f"[t={t:7.2f}s] skip (clip fail: {e})", flush=True)
            t += args.step; continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inputs = processor(
                    text=[text_prompt],
                    videos=[str(clip)],
                    video_backend="codec",
                    codec_config=CODEC_CONFIG,
                    max_pixels=CODEC_MAX_PIXELS,
                    return_tensors="pt",
                    padding=False,
                )
        except Exception as e:
            print(f"[t={t:7.2f}s] skip (codec fail: {e})", flush=True)
            t += args.step; continue

        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        gate_pos = torch.tensor([inputs["input_ids"].size(1) - 1], device=device)

        with torch.no_grad():
            out = model.base(**inputs, output_hidden_states=True, return_dict=True)
            h = out.hidden_states[-1]
            g_in = h[torch.arange(h.size(0), device=device), gate_pos]
            # gate is fp32; base hidden is bf16
            g_logits = model.gate(g_in.float())
            probs = F.softmax(g_logits, dim=-1).squeeze(0).tolist()

        respond_prob = probs[1]
        if respond_prob >= args.gate_thresh:
            with torch.no_grad():
                gen = model.base.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            new_ids = gen[0, inputs["input_ids"].size(1):]
            caption = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            events.append((t, caption))
            print(f"[t={t:7.2f}s gate={respond_prob:.2f}] {caption}", flush=True)
        else:
            print(f"[t={t:7.2f}s gate={respond_prob:.2f}] (silence)", flush=True)

        t += args.step

    # Optional comparison with ground truth
    if gts:
        print("\n[demo] === ground-truth timeline ===")
        for t, text in gts:
            if args.t_start <= t <= args.t_end:
                print(f"  GT [t={t:7.2f}s] {text}")

    # Optional subtitle burn-in
    if args.output_video and events:
        srt = args.output_video + ".srt"
        write_srt(events, srt, args.t_start)
        print(f"\n[demo] writing video with burnt subtitles -> {args.output_video}", flush=True)
        burn_subs(args.video, srt, args.t_start, args.t_end, args.output_video)
        print(f"[demo] done. srt at {srt}", flush=True)


if __name__ == "__main__":
    main()
