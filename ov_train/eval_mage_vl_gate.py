"""Evaluate the optional StreamMind gate shipped with Mage-VL-Base.

Example (8 GPUs):

    MATCHTIME_ROOT=/data/kaichen/data/MatchTime \
    SOCCER_MANIFEST_DIR=/data/kaichen/data/MatchTime/manifests \
    ONLINE_CODEC_CACHE_DIR=/data/kaichen/data/MatchTime/codec_cache_p16_valid \
    torchrun --nproc_per_node 8 -m ov_train.eval_mage_vl_gate \
      --model Mage-VL/Mage-VL-Base

The main Mage-VL forward/generate path is unchanged. Gate weights are loaded
lazily from ``streammind_gate.safetensors`` only when this evaluator invokes
``streammind_gate_forward_segments``.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from ov_train.stream_dataset import DataArguments, LazySupervisedDataset
from ov_train.train import DataCollatorForstreamDataset, _resolve_image_processor


def relaxed_correct(labels, predictions, tolerance):
    matches = torch.zeros_like(labels, dtype=torch.bool)
    for index in range(len(labels)):
        start = max(0, index - tolerance)
        end = min(len(labels), index + tolerance + 1)
        if labels[index] in predictions[start:end]:
            matches[index] = True
    return matches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Mage-VL/Mage-VL-Base")
    parser.add_argument("--data_type", default="valid")
    parser.add_argument("--tolerance_frames", type=int, default=2)
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(local_rank).eval()

    data_args = DataArguments()
    data_args.soccer_dataset = True
    data_args.soccer_dataset_train_cls = True
    data_args.soccer_dataset_train_llm = False
    data_args.video_backend = "codec"
    data_args.data_type = args.data_type
    data_args.video_processor = _resolve_image_processor(processor)
    data_args.image_processor = data_args.video_processor
    data_args.processor = processor
    data_args.is_multimodal = True
    data_args.cur_fps = 2

    dataset = LazySupervisedDataset(data_path="", tokenizer=tokenizer, data_args=data_args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=DataCollatorForstreamDataset(tokenizer=tokenizer),
    )

    trigger_scores = []
    timval_scores = []
    output_path = ""
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"rank_{dist.get_rank():02d}.jsonl")
    for half, inputs in enumerate(loader, 1):
        if (half - 1) % dist.get_world_size() != dist.get_rank():
            continue
        segments = inputs["images"][0]
        for segment in segments:
            for key in ("pixel_values", "image_grid_thw", "patch_positions"):
                if key in segment:
                    segment[key] = segment[key].to(local_rank)
        with torch.no_grad():
            logits = model.streammind_gate_forward_segments(segments)[0]

        boundaries = torch.tensor(
            [segment["image_grid_thw"][:, 0].sum() for segment in segments]
        ).cumsum(0).to(logits.device)
        labels = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
        labels[boundaries - 1] = 1
        probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]
        predictions = logits.argmax(dim=-1)
        relaxed = relaxed_correct(labels, predictions, args.tolerance_frames)

        if output_path:
            record = {
                "half": half - 1,
                "tolerance": args.tolerance_frames,
                "labels": labels.cpu().tolist(),
                "predictions": predictions.cpu().tolist(),
                "response_probabilities": probabilities.cpu().tolist(),
            }
            with open(output_path, "a") as file:
                file.write(json.dumps(record) + "\n")

        trigger_scores.append(relaxed.float().mean())
        false_positives = (((labels == 0) & (predictions == 1)) & ~relaxed).sum()
        false_negatives = (((labels == 1) & (predictions == 0)) & ~relaxed).sum()
        silence_rate = 1 - false_positives / (labels == 0).sum().clamp_min(1)
        speak_rate = 1 - false_negatives / (labels == 1).sum().clamp_min(1)
        timval_scores.append(silence_rate * speak_rate)

    totals = torch.tensor(
        [sum(trigger_scores), sum(timval_scores), len(trigger_scores)],
        dtype=torch.float64,
        device=local_rank,
    )
    dist.all_reduce(totals)
    if dist.get_rank() == 0:
        print(f"TriggerAcc: {(totals[0] / totals[2]).item():.6f}")
        print(f"TimVal:     {(totals[1] / totals[2]).item():.6f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
