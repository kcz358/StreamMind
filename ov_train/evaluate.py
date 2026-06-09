"""SoccerNet evaluation for paper-faithful StreamMind OneVision.

Port of the paper's `streammind/eval/inference_video_ego4d_stream_parallel_new.py`
with two eval types:

  --eval_type cls -> gate metrics (TriggerAcc, TimeVal)
    relaxed_correct within ±tolerance_frames of the target slot.

  --eval_type llm -> teacher-forced caption metrics (PPL, Correctness, Fluency,
    plus dumped caption text for offline BLEU/CIDEr).

Both modes reuse LazySupervisedDataset + DataCollatorForstreamDataset (same as
training), so the eval distribution exactly matches the train distribution.

Usage (inside dev pod, single GPU):
  python -m ov_train.evaluate \\
    --resume_from /data/kaichen/StreamMind/paper_stage2_codec_.../checkpoint-470 \\
    --eval_type cls \\
    --tolerance_frames 2 \\
    --max_halves -1

  python -m ov_train.evaluate \\
    --resume_from /data/kaichen/StreamMind/paper_stage1_codec_.../checkpoint-94 \\
    --eval_type llm \\
    --caption_csv /tmp/eval_captions.csv \\
    --max_halves -1
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streammind.constants import IGNORE_INDEX
from ov_train.onevision_stream import OneVisionStreamForCausalLM
from ov_train.stream_dataset import DataArguments, LazySupervisedDataset
from ov_train.train import DataCollatorForstreamDataset, _resolve_image_processor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/data/kaichen/LLaVA-OneVision-1.5-RL/pretrained/LLaVA-OneVision-2-8B-Instruct")
    p.add_argument("--resume_from", required=True, help="stage1 or stage2 checkpoint dir")
    p.add_argument("--eval_type", choices=["cls", "llm"], required=True)
    p.add_argument("--data_type", default="valid")
    p.add_argument("--max_halves", type=int, default=-1, help="cap halves; -1 = all")
    p.add_argument("--tolerance_frames", type=int, default=2, help="cls: relaxed_correct tolerance")
    p.add_argument("--caption_csv", default="", help="llm: write per-turn (pred, target) rows")
    return p.parse_args()


def relaxed_correct(eos_labels: torch.Tensor, pred_labels: torch.Tensor, N: int) -> torch.Tensor:
    """Per-position match in a ±N window. Direct port of paper line 128."""
    matches = torch.zeros_like(eos_labels, dtype=torch.bool)
    for i in range(len(eos_labels)):
        start = max(0, i - N)
        end = min(len(eos_labels), i + N + 1)
        if eos_labels[i] in pred_labels[start:end]:
            matches[i] = True
    return matches


def build(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    image_processor = _resolve_image_processor(processor)

    data_args = DataArguments()
    data_args.soccer_dataset = True
    if args.eval_type == "cls":
        data_args.soccer_dataset_train_cls = True
        data_args.soccer_dataset_train_llm = False
    else:
        data_args.soccer_dataset_train_cls = False
        data_args.soccer_dataset_train_llm = True
    data_args.video_backend = "codec"
    data_args.data_type = args.data_type
    data_args.video_processor = image_processor
    data_args.image_processor = image_processor
    data_args.processor = processor
    data_args.is_multimodal = True
    data_args.cur_fps = 2

    print(f"[eval] loading base model: {args.model_path}")
    model = OneVisionStreamForCausalLM(args.model_path)
    model.add_streammind_special_tokens(tokenizer)

    print(f"[eval] loading ckpt: {args.resume_from}")
    sd = load_file(os.path.join(args.resume_from, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[eval]   loaded={len(sd) - len(unexpected)} missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device=device, dtype=torch.bfloat16).eval()
    dataset = LazySupervisedDataset(data_path="", tokenizer=tokenizer, data_args=data_args)
    collator = DataCollatorForstreamDataset(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collator)
    return tokenizer, model, dataset, loader


def eval_cls(args, model, loader):
    """Paper line 238-315: TriggerAcc + TimeVal."""
    trigger_acc_list = []
    time_val_list = []
    n_halves = 0
    for batch_idx, inputs in enumerate(loader):
        if args.max_halves >= 0 and n_halves >= args.max_halves:
            break
        n_halves += 1
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(model.model.ov_model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        if not isinstance(outputs, tuple):
            print(f"[half {n_halves}] unexpected output type {type(outputs)}")
            continue
        cls_output, cls_label = outputs
        logits = cls_output.logits[..., :-1, :]
        labels = cls_label[..., 1:].to(logits.device)

        logits_flat = logits.reshape(-1, logits.shape[-1])
        labels_flat = labels.reshape(-1)
        target_mask = labels_flat != IGNORE_INDEX
        eos_logits = logits_flat[target_mask]
        eos_labels = labels_flat[target_mask]
        probs = torch.softmax(eos_logits.float(), dim=-1)
        pred_labels = probs.argmax(dim=-1)

        relaxed = relaxed_correct(eos_labels, pred_labels, args.tolerance_frames)
        correct = relaxed.sum().item()
        trigger_acc = correct / (eos_labels.numel() + 1e-9)
        trigger_acc_list.append(trigger_acc)

        false_positives = (((eos_labels == 0) & (pred_labels == 1)) & ~relaxed).sum().item()
        total_negatives = (eos_labels == 0).sum().item()
        true_positive_rate = 1 - false_positives / (total_negatives + 1e-9)

        false_negatives = (((eos_labels == 1) & (pred_labels == 0)) & ~relaxed).sum().item()
        total_positives = (eos_labels == 1).sum().item()
        true_negative_rate = 1 - false_negatives / (total_positives + 1e-9)

        time_val = true_positive_rate * true_negative_rate
        time_val_list.append(time_val)

        print(f"[half {n_halves}] target_slots={eos_labels.numel()} "
              f"TriggerAcc={trigger_acc:.4f} TimeVal={time_val:.4f} "
              f"(TPR={true_positive_rate:.4f} TNR={true_negative_rate:.4f})")

    print(f"\n[eval cls, {n_halves} halves]")
    print(f"  TriggerAcc (mean): {sum(trigger_acc_list) / max(len(trigger_acc_list), 1):.4f}")
    print(f"  TimeVal    (mean): {sum(time_val_list) / max(len(time_val_list), 1):.4f}")


def eval_llm(args, tokenizer, model, loader):
    """Paper line 162-237: PPL + Correctness + Fluency, plus caption csv."""
    if args.caption_csv:
        os.makedirs(os.path.dirname(args.caption_csv) or ".", exist_ok=True)
        with open(args.caption_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["pred", "target"])

    video_lm_ppls = []
    video_lm_correctness = []
    video_lm_fluency = []   # paper's "lm_correctness_token" = avg correct tokens per caption
    video_token_total = []
    n_halves = 0
    for batch_idx, inputs in enumerate(loader):
        if args.max_halves >= 0 and n_halves >= args.max_halves:
            break
        n_halves += 1
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(model.model.ov_model.device)
        inputs["llm_eval"] = True

        with torch.no_grad():
            output, labels = model(**inputs)
        labels = labels.to(output.logits.device)
        logit = output.logits[0]

        # paper line 190: turns are positions where label == 2 (Mistral EOS).
        # Qwen3 EOS is tokenizer.eos_token_id (151645). Use that here.
        eos_id = tokenizer.eos_token_id
        turns = (labels == eos_id).nonzero(as_tuple=True)[1].tolist()
        if not turns:
            print(f"[half {n_halves}] no turn boundaries (eos={eos_id})")
            continue

        start_turns = [-1] + turns[:-1]
        lm_ppls, lm_correctness, token_num, correct_token_num = [], [], [], []
        for idx, turn in enumerate(turns):
            turn_logit = logit[start_turns[idx] + 1:turn + 1]
            turn_label = labels[0][start_turns[idx] + 1:turn + 1]
            turn_label = turn_label[1:]
            turn_logit = turn_logit[:-1]
            mask = turn_label != IGNORE_INDEX
            turn_logit = turn_logit[mask]
            turn_label = turn_label[mask]
            if turn_label.numel() == 0:
                continue
            pred_ids = turn_logit.argmax(dim=-1)
            pred_text = tokenizer.batch_decode(pred_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
            target_text = tokenizer.batch_decode(turn_label.unsqueeze(0), skip_special_tokens=True)[0].strip()
            if args.caption_csv:
                with open(args.caption_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([pred_text, target_text])

            ppl = torch.nn.functional.cross_entropy(turn_logit.float(), turn_label).exp()
            lm_ppls.append(ppl)
            correct = (pred_ids == turn_label).sum()
            correct_token_num.append(correct)
            token_num.append(turn_label.numel())
            lm_correctness.append(correct.float() / turn_label.numel())

        if not lm_ppls:
            continue
        video_lm_ppls.append(sum(lm_ppls) / len(lm_ppls))
        video_lm_correctness.append(sum(lm_correctness) / len(lm_correctness))
        video_lm_fluency.append(sum(correct_token_num) / len(correct_token_num))
        video_token_total.append(sum(token_num) / len(token_num))
        print(f"[half {n_halves}] turns={len(turns)} "
              f"PPL={float(video_lm_ppls[-1]):.4f} "
              f"Correctness={float(video_lm_correctness[-1]):.4f} "
              f"Fluency={float(video_lm_fluency[-1]):.2f} "
              f"AvgTokens={float(video_token_total[-1]):.2f}")

    n = max(len(video_lm_ppls), 1)
    print(f"\n[eval llm, {n_halves} halves]")
    print(f"  PPL         (mean): {float(sum(video_lm_ppls) / n):.4f}")
    print(f"  Correctness (mean): {float(sum(video_lm_correctness) / n):.4f}")
    print(f"  Fluency     (mean): {float(sum(video_lm_fluency) / n):.2f}")
    print(f"  AvgTokens   (mean): {float(sum(video_token_total) / n):.2f}")
    if args.caption_csv:
        print(f"  captions csv: {args.caption_csv}")


def main():
    args = parse_args()
    device = "cuda"
    tokenizer, model, dataset, loader = build(args, device)
    if args.eval_type == "cls":
        eval_cls(args, model, loader)
    else:
        eval_llm(args, tokenizer, model, loader)


if __name__ == "__main__":
    main()
