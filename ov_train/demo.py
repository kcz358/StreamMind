"""Audit-only streaming demo for paper-faithful StreamMind OneVision.

For a single MatchTime half, slide an 8-second window every `--step` seconds.
At every step:
  1. ffmpeg extract subclip → codec backend → ViT → Mamba EPFE → ClsNet
  2. Print [t_start, p_silence, p_response, predicted_class]
  3. If predicted_class == 1 (response), call base LLM .generate() to emit a one-line caption

This is the paper's `stream_generate_demo` path:
  prepare_inputs_labels_for_multimodal_score_stream_inference_demo
    -> encode_images_or_videos_score_cls_inference_allframe_demo  (ViT + Mamba + ClsNet)
    -> if cls_pred == 1: super().generate(inputs_embeds=...)

Run inside the dev pod:
  python -m ov_train.demo \\
    --resume_from /data/kaichen/StreamMind/paper_stage2_codec_.../checkpoint-94 \\
    --video       /data/kaichen/data/MatchTime/features_video/.../1_224p.mkv \\
    --captions    /data/kaichen/data/MatchTime/dataset/MatchTime/valid/.../Labels-caption.json \\
    --half 1 --t_start 0 --t_end 300 --step 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ov_train.codec_utils import extract_subclip
from ov_train.onevision_stream import OneVisionStreamForCausalLM


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
    p.add_argument("--window", type=float, default=8.0, help="window length in seconds")
    p.add_argument("--audit_only", action="store_true", help="skip LLM generate, only print gate")
    p.add_argument("--clip_cache", default="/tmp/demo_clips")
    return p.parse_args()


def _parse_gt(captions_path: str, half: int):
    """Return list of (t_sec, text) for ground-truth captions in given half."""
    if not captions_path:
        return []
    data = json.load(open(captions_path))
    gts = []
    for ann in data.get("annotations", []):
        gameTime = ann.get("gameTime", "")
        try:
            h_str, mmss = gameTime.split(" - ")
            h = int(h_str.strip().split(" ")[0])
            if h != half:
                continue
            m, s = mmss.strip().split(":")
            t = int(m) * 60 + int(s)
        except Exception:
            continue
        gts.append((t, ann.get("anonymized", ann.get("description", ""))))
    return sorted(gts)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.clip_cache, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"[demo] loading base model: {args.model_path}")
    model = OneVisionStreamForCausalLM(args.model_path)
    model.add_streammind_special_tokens(tokenizer)

    print(f"[demo] loading stage2 ckpt: {args.resume_from}")
    sd = load_file(os.path.join(args.resume_from, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[demo]   loaded={len(sd) - len(unexpected)} missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device=device, dtype=torch.bfloat16).eval()

    # silence / response token ids (set on the Mamba projector during init)
    silence_id = model.model.mm_projector.silence_token_id
    response_id = model.model.mm_projector.response_token_id
    print(f"[demo] silence_id={silence_id} response_id={response_id}")

    gt_list = _parse_gt(args.captions, args.half)
    print(f"[demo] ground-truth captions in half {args.half}: {len(gt_list)}")

    # Sliding window
    t = args.t_start
    n_silence = 0
    n_response = 0
    p_resp_history = []
    while t + args.window <= args.t_end:
        clip_path = os.path.join(args.clip_cache, f"clip_{int(t*100):08d}_{int(args.window*100):04d}.mp4")
        try:
            extract_subclip(args.video, t, args.window, clip_path)
            enc = processor(
                text=["<video>"], videos=[clip_path],
                video_backend="codec", return_tensors="pt", padding=False,
            )
            codec_dict = {
                "pixel_values": enc["pixel_values"].to(device=device, dtype=torch.bfloat16),
                "image_grid_thw": enc["image_grid_thw"].to(device=device),
                "patch_positions": enc["patch_positions"].to(device=device),
            }
        except Exception as e:
            print(f"[t={t:6.1f}] codec failed: {e}")
            t += args.step
            continue

        with torch.no_grad():
            # Direct encoder-side audit: bypass prepare_inputs* and call the
            # encode helper that the inference_demo path uses internally. This
            # gives us the raw cls_feature softmax without the LLM splice.
            past_frames = getattr(model, "_demo_past_frames", None)
            interval_id_list = getattr(model, "_demo_interval_ids", [])
            X_features, cls_feature, new_frames, interval_id = (
                model.encode_images_or_videos_score_cls_inference_allframe_demo(
                    codec_dict, past_frames, frames_features_shape=interval_id_list
                )
            )
            model._demo_past_frames = new_frames
            interval_id_list.append(interval_id)
            model._demo_interval_ids = interval_id_list

            cls_probs = torch.softmax(cls_feature.float().flatten(), dim=-1).tolist()
            cls_pred_int = int(torch.tensor(cls_probs).argmax().item())

        if cls_pred_int == 0:
            n_silence += 1
            print(f"[t={t:6.1f}] cls=0 p_sil={cls_probs[0]:.3f} p_resp={cls_probs[1]:.3f}")
        else:
            n_response += 1
            print(f"[t={t:6.1f}] cls=1 p_sil={cls_probs[0]:.3f} p_resp={cls_probs[1]:.3f}")

        # GT alignment: print any GT caption that lies in [t, t+step)
        for (g_t, g_txt) in gt_list:
            if t <= g_t < t + args.step:
                print(f"    GT @ {g_t}s: {g_txt}")

        t += args.step

    print(f"\n[demo] done. silence={n_silence} response={n_response}")


if __name__ == "__main__":
    main()
