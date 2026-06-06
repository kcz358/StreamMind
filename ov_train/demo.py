"""Paper-faithful audit demo for StreamMind OneVision.

This script mirrors the paper's `eval_type=cls` evaluation path in
`streammind/eval/inference_video_ego4d_stream_parallel_new.py`:

  1. Build LazySupervisedDataset with soccer_dataset_train_cls=True (same as
     training).
  2. Forward one half through the model with cls_inference=True (paper Path-B,
     same distribution as cls_training).
  3. Extract logits at the target slots (cls_label != IGNORE_INDEX) and
     argmax to get per-segment silence/response predictions.
  4. Compare to ground-truth cls_label and print accuracy.

Run inside the dev pod (single GPU is enough):
  python -m ov_train.demo \\
    --resume_from /data/kaichen/StreamMind/paper_stage2_codec_.../checkpoint-94 \\
    --max_halves 1
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streammind.constants import IGNORE_INDEX
from ov_train.onevision_stream import OneVisionStreamForCausalLM
from ov_train.stream_dataset import (
    DataArguments,
    LazySupervisedDataset,
)
from ov_train.train import DataCollatorForstreamDataset, _resolve_image_processor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/data/kaichen/LLaVA-OneVision-1.5-RL/pretrained/LLaVA-OneVision-2-8B-Instruct")
    p.add_argument("--resume_from", required=True, help="stage2 checkpoint dir")
    p.add_argument("--data_type", default="valid", help="train|valid")
    p.add_argument("--max_halves", type=int, default=1)
    return p.parse_args()


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

    silence_id = model.model.mm_projector.silence_token_id
    response_id = model.model.mm_projector.response_token_id
    print(f"[demo] silence_id={silence_id} response_id={response_id}")

    dataset = LazySupervisedDataset(data_path="", tokenizer=tokenizer, data_args=data_args)
    collator = DataCollatorForstreamDataset(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collator)

    total_silence_correct = 0
    total_silence = 0
    total_response_correct = 0
    total_response = 0

    n_done = 0
    for batch_idx, inputs in enumerate(loader):
        if n_done >= args.max_halves:
            break
        n_done += 1
        # move tensors to device (collator keeps lists for images)
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # `outputs` is whatever model.forward returns for cls path. Per
        # `OneVisionStreamForCausalLM.forward`, when cls_output is not None it
        # returns cls_output directly (a CausalLMOutputWithPast-shaped object).
        # The paper eval then unpacks (outputs, labels) via:
        #   outputs, labels = model(**inputs)
        # We replicate this and grab the `labels` from outputs.logits / .loss.
        # In paper's projector, the cls path returns (cls_output, cls_label).
        # Our train.py forward wraps that to single object; we re-call the
        # encoder directly to grab both pieces.
        if isinstance(outputs, tuple):
            cls_output, cls_label = outputs
        else:
            cls_output = outputs
            cls_label = inputs.get("labels", None)
            if cls_label is None:
                # The cls path's label is built inside projector; recover via
                # a private re-call.
                raise RuntimeError("could not recover cls_label from model output")

        logits = cls_output.logits  # [B, T, vocab=2]
        logits = logits[..., :-1, :]
        labels = cls_label[..., 1:].to(logits.device)

        # Flatten per paper eval (line 263-279): treat the (b, t=2) batching by
        # concatenating sequences and picking target slots where label != IGNORE.
        logits_flat = logits.reshape(-1, logits.shape[-1])
        labels_flat = labels.reshape(-1)
        target_mask = labels_flat != IGNORE_INDEX
        eos_logits = logits_flat[target_mask]
        eos_labels = labels_flat[target_mask]

        probs = torch.softmax(eos_logits.float(), dim=-1)
        preds = probs.argmax(dim=-1)

        # paper labels: 0 = silence, 1 = response (these are cls_net vocab ids)
        silence_mask = eos_labels == 0
        response_mask = eos_labels == 1
        silence_correct = (preds[silence_mask] == 0).sum().item()
        response_correct = (preds[response_mask] == 1).sum().item()
        n_silence = silence_mask.sum().item()
        n_response = response_mask.sum().item()

        total_silence += n_silence
        total_silence_correct += silence_correct
        total_response += n_response
        total_response_correct += response_correct

        print(f"[half {n_done}] target_slots={target_mask.sum().item()} "
              f"silence_acc={silence_correct}/{n_silence} "
              f"response_acc={response_correct}/{n_response} "
              f"p_resp_mean={probs[:, 1].mean().item():.3f}")

    print(f"\n[demo] overall:")
    print(f"  silence acc: {total_silence_correct}/{total_silence} = {total_silence_correct / max(total_silence, 1):.3f}")
    print(f"  response acc: {total_response_correct}/{total_response} = {total_response_correct / max(total_response, 1):.3f}")


if __name__ == "__main__":
    main()
