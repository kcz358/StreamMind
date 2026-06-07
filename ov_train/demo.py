"""Paper-faithful streaming demo for StreamMind OneVision.

Two-pass:
  Pass 1 (cls audit): Forward whole half through cls_inference path (same as
    paper eval_type=cls) -> per-segment gate decisions + GT labels.
  Pass 2 (caption generate, only on fire segments): For each segment where the
    gate predicted response=1, build a single-video stage1 prompt and call
    LLM.generate() to produce the actual commentary text.

This avoids paper's buggy `stream_generate_demo` (which uses cls_demo single-
frame path) and stays aligned with our pair-based ClsNet training.

Usage inside dev pod:
  python -m ov_train.demo \\
    --resume_from /data/kaichen/StreamMind/paper_stage2_codec_.../checkpoint-470 \\
    --data_type valid --max_halves 1
"""
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streammind.constants import IGNORE_INDEX, MMODAL_TOKEN_INDEX
from streammind.mm_utils import tokenizer_MMODAL_token
from ov_train.onevision_stream import OneVisionStreamForCausalLM
from ov_train.stream_dataset import (
    DataArguments,
    LazySupervisedDataset,
)
from ov_train.train import DataCollatorForstreamDataset, _resolve_image_processor


SYS_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
USER_PREFIX = "Please describe the video content in detail based on the provided information."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/data/kaichen/LLaVA-OneVision-1.5-RL/pretrained/LLaVA-OneVision-2-8B-Instruct")
    p.add_argument("--resume_from", required=True, help="stage2 checkpoint dir")
    p.add_argument("--data_type", default="valid")
    p.add_argument("--max_halves", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--max_generate_segs", type=int, default=20, help="cap LLM.generate calls per half")
    return p.parse_args()


@torch.no_grad()
def generate_for_segment(model, tokenizer, segment_video_dict, device, max_new_tokens):
    """Run LLM.generate on a single segment's codec dict and return decoded text."""
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": USER_PREFIX + "<video>\n"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_TOKEN_INDEX["VIDEO"], return_tensors="pt").unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    images = [[segment_video_dict], ["video"]]

    (
        new_input_ids,
        new_attn,
        _pkv,
        inputs_embeds,
        _labels,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids, attention_mask, None, None, images,
    )

    if inputs_embeds is None:
        return "(no inputs_embeds)"

    out_ids = model.model.language_model.generate(
        inputs_embeds=inputs_embeds.to(dtype=torch.bfloat16),
        attention_mask=new_attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    return text


def main():
    args = parse_args()
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    image_processor = _resolve_image_processor(processor)

    data_args = DataArguments()
    data_args.soccer_dataset = True
    data_args.soccer_dataset_train_cls = True
    data_args.soccer_dataset_train_llm = False
    data_args.video_backend = "codec"
    data_args.data_type = args.data_type
    data_args.video_processor = image_processor
    data_args.image_processor = image_processor
    data_args.processor = processor
    data_args.is_multimodal = True
    data_args.cur_fps = 2

    print(f"[demo] loading base model: {args.model_path}")
    model = OneVisionStreamForCausalLM(args.model_path)
    model.add_streammind_special_tokens(tokenizer)

    print(f"[demo] loading stage2 ckpt: {args.resume_from}")
    sd = load_file(os.path.join(args.resume_from, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[demo]   loaded={len(sd) - len(unexpected)} missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device=device, dtype=torch.bfloat16).eval()

    dataset = LazySupervisedDataset(data_path="", tokenizer=tokenizer, data_args=data_args)
    collator = DataCollatorForstreamDataset(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collator)

    n_done = 0
    for batch_idx, inputs in enumerate(loader):
        if n_done >= args.max_halves:
            break
        n_done += 1
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        # ---- Pass 1: cls audit on whole half ----
        with torch.no_grad():
            outputs = model(**inputs)
        if isinstance(outputs, tuple):
            cls_output, cls_label = outputs
        else:
            print(f"[half {n_done}] unexpected model output type: {type(outputs)}")
            continue
        logits = cls_output.logits[..., :-1, :]
        labels_shift = cls_label[..., 1:].to(logits.device)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        labels_flat = labels_shift.reshape(-1)
        target_mask = labels_flat != IGNORE_INDEX
        eos_logits = logits_flat[target_mask]
        eos_labels = labels_flat[target_mask]
        probs = torch.softmax(eos_logits.float(), dim=-1)
        preds = probs.argmax(dim=-1)
        fire_idx = (preds == 1).nonzero(as_tuple=True)[0].tolist()
        gt_idx = (eos_labels == 1).nonzero(as_tuple=True)[0].tolist()
        print(f"[half {n_done}] target_slots={target_mask.sum().item()} "
              f"gate_fired={len(fire_idx)} gt_response={len(gt_idx)}")

        # ---- Pass 2: for each fire, find which segment it belongs to and
        # generate caption ----
        # Each segment has its own (frame, target) pairs. The fire indices tell
        # us which target slot (after pair-flattening) predicted response. We
        # need the matching original segment index.
        # In Path B, each segment of N frames produces N pair rows (the last
        # one has label==1, the others label==0). So a fire at the *last* row
        # of a segment block corresponds to that segment's caption boundary.
        # Easiest: any segment whose last-pair index is in fire_idx -> generate
        # for that segment.
        # We don't have direct segment indices here; reconstruct by walking
        # gt_idx (each gt_idx item == last pair row of one segment).
        segment_video_dicts = inputs.get("images")  # [[codec_dict_seg_0, ..., codec_dict_seg_M], ["video"]]
        if segment_video_dicts is None or len(segment_video_dicts[0]) == 0:
            continue
        seg_codec_list = segment_video_dicts[0]

        n_generated = 0
        for seg_idx, last_pair_pos in enumerate(gt_idx):
            if n_generated >= args.max_generate_segs:
                break
            fired = last_pair_pos in fire_idx
            if not fired:
                continue
            if seg_idx >= len(seg_codec_list):
                break
            seg_dict = seg_codec_list[seg_idx]
            if not isinstance(seg_dict, dict):
                continue
            # move codec tensors to device
            seg_dict_d = {
                k: (v.to(device=device, dtype=torch.bfloat16) if k == "pixel_values" else v.to(device))
                for k, v in seg_dict.items()
            }
            try:
                text = generate_for_segment(model, tokenizer, seg_dict_d, device, args.max_new_tokens)
            except Exception as e:
                text = f"(generate failed: {e})"
            n_generated += 1
            print(f"  seg {seg_idx} fired -> {text[:200]}")

        print(f"[half {n_done}] generated {n_generated} captions.")


if __name__ == "__main__":
    main()
