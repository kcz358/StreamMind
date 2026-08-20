import argparse
import json
import math
import os
import time

import decord
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed


SYSTEM_PROMPT = """You are a real-time video streaming assistant observing a continuous camera feed frame by frame. The last frame represents the current moment.
## Action Format
At every inference step you MUST choose exactly one of the following two actions:
**Stay silent** - output ONLY:
</silence>
Choose this when nothing noteworthy has changed in the scene, no user query is pending, or there is nothing useful to say.
**Speak** - output the token followed by a concise reply:
</response> Your reply here.
Choose this when you observe something worth reporting or a significant state change, or when you can answer a user question based on available evidence."""
QUERY = "Watch this soccer match and alert me whenever a noteworthy event occurs."


def patch_embed_forward(self, hidden_states):
    target_dtype = self.proj.weight.dtype
    with torch.amp.autocast(device_type="cuda", enabled=False):
        hidden_states = F.linear(
            hidden_states.float(),
            self.proj.weight.view(self.embed_dim, -1).float(),
            self.proj.bias.float() if self.proj.bias is not None else None,
        )
    return hidden_states.to(target_dtype)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="jdopensource/JoyAI-VL-Interaction-Preview")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chunk_seconds", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_halves", type=int, default=-1)
    parser.add_argument("--max_seconds", type=int, default=-1)
    parser.add_argument("--output", default="")
    parser.add_argument("--worker_group", type=int, default=0)
    parser.add_argument("--worker_groups", type=int, default=1)
    parser.add_argument("--progress_dir", default="")
    parser.add_argument("--tolerance_seconds", type=int, default=0)
    return parser.parse_args()


def event_times(row):
    times = []
    for annotation in row.annotations:
        game_time = annotation.get("gameTime", "")
        if " - " not in game_time:
            continue
        head, value = game_time.split(" - ", 1)
        if int(head.strip().split()[0]) != int(row.half):
            continue
        minute, second = value.strip().split(":")
        times.append(int(minute) * 60 + int(second))
    return sorted(set(times))


def load_frames(reader, fps, seconds):
    indices = [min(len(reader) - 1, max(0, round(second * fps))) for second in seconds]
    arrays = reader.get_batch(indices).asnumpy()
    return [Image.fromarray(array).convert("RGB") for array in arrays]


def relaxed_correct(labels, predictions, tolerance):
    matches = torch.zeros_like(labels, dtype=torch.bool)
    for index in range(len(labels)):
        start = max(0, index - tolerance)
        end = min(len(labels), index + tolerance + 1)
        if labels[index] in predictions[start:end]:
            matches[index] = True
    return matches


def predict_chunks(model, processor, reader, fps, chunks, device):
    texts = []
    images = []
    for seconds, labels in chunks:
        images.extend(load_frames(reader, fps, seconds))
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for index, (second, label) in enumerate(zip(seconds, labels)):
            text = f"<{second:.1f} seconds>"
            if index == 0:
                text = f"[User Query (IMPORTANT - follow this instruction)]\n{QUERY}\n{text}"
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}, {"type": "image"}],
                }
            )
            action = "</response> A noteworthy soccer event occurred." if label else "</silence>"
            messages.append({"role": "assistant", "content": action})
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    silence_id = processor.tokenizer.convert_tokens_to_ids("</silence>")
    response_id = processor.tokenizer.convert_tokens_to_ids("</response>")
    assistant_prefix = processor.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    with torch.inference_mode():
        hidden_states = model.model(**inputs).last_hidden_state
        action_weights = model.lm_head.weight[[silence_id, response_id]]
    predictions = []
    probabilities = []
    for batch_index, (_, labels) in enumerate(chunks):
        input_ids = inputs["input_ids"][batch_index]
        candidates = ((input_ids == silence_id) | (input_ids == response_id)).nonzero().flatten()
        positions = torch.tensor(
            [
                position
                for position in candidates.tolist()
                if input_ids[position - len(assistant_prefix) : position].tolist() == assistant_prefix
            ],
            device=input_ids.device,
        )
        if len(positions) != len(labels):
            raise RuntimeError(f"action positions={len(positions)} labels={len(labels)}")
        action_logits = hidden_states[batch_index, positions - 1] @ action_weights.T
        predictions.append(action_logits.argmax(dim=-1).cpu())
        probabilities.append(torch.softmax(action_logits.float(), dim=-1)[:, 1].cpu())
    return predictions, probabilities


def main():
    args = parse_args()
    Qwen3VLVisionPatchEmbed.forward = patch_embed_forward
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device).eval()

    frame = pd.read_parquet(args.manifest)
    if args.max_halves >= 0:
        frame = frame.iloc[: args.max_halves]
    trigger_sum = 0.0
    timeval_sum = 0.0
    half_count = 0
    started = time.time()
    global_rank = args.worker_group * dist.get_world_size() + dist.get_rank()
    global_world_size = args.worker_groups * dist.get_world_size()
    progress_path = ""
    if args.progress_dir:
        os.makedirs(args.progress_dir, exist_ok=True)
        progress_path = os.path.join(args.progress_dir, f"worker_{global_rank:02d}.jsonl")

    for half_index, row in frame.iterrows():
        if half_index % global_world_size != global_rank:
            continue
        events = event_times(row)
        if not events:
            continue
        reader = decord.VideoReader(row.video_path, ctx=decord.cpu(0), num_threads=1)
        fps = float(reader.get_avg_fps())
        end = min(events[-1], math.floor((len(reader) - 1) / fps))
        if args.max_seconds >= 0:
            end = min(end, args.max_seconds)
        labels = torch.zeros(end + 1, dtype=torch.long)
        labels[[event for event in events if event <= end]] = 1
        chunks = []
        for start in range(0, end + 1, args.chunk_seconds):
            stop = min(end + 1, start + args.chunk_seconds)
            chunks.append((list(range(start, stop)), labels[start:stop]))
        predictions = []
        probabilities = []
        for start in range(0, len(chunks), args.batch_size):
            batch_predictions, batch_probabilities = predict_chunks(
                model, processor, reader, fps, chunks[start : start + args.batch_size], device
            )
            predictions.extend(batch_predictions)
            probabilities.extend(batch_probabilities)
        predictions = torch.cat(predictions)
        probabilities = torch.cat(probabilities)
        correct = relaxed_correct(labels, predictions, args.tolerance_seconds)
        trigger_sum += correct.float().mean().item()
        false_positives = (((labels == 0) & (predictions == 1)) & ~correct).sum()
        false_negatives = (((labels == 1) & (predictions == 0)) & ~correct).sum()
        silence_rate = 1 - false_positives / (labels == 0).sum().clamp_min(1)
        speak_rate = 1 - false_negatives / (labels == 1).sum().clamp_min(1)
        timeval_sum += (silence_rate * speak_rate).item()
        half_count += 1
        progress = {
            "worker": global_rank,
            "half": int(half_index),
            "local_done": half_count,
            "trigger_sum": trigger_sum,
            "timeval_sum": timeval_sum,
            "elapsed": time.time() - started,
        }
        line = json.dumps(progress)
        print(line, flush=True)
        if progress_path:
            with open(progress_path, "a") as file:
                file.write(line + "\n")
            record_path = os.path.join(args.progress_dir, f"records_{global_rank:02d}.jsonl")
            record = {
                "half": int(half_index),
                "tolerance": args.tolerance_seconds,
                "labels": labels.tolist(),
                "predictions": predictions.tolist(),
                "response_probabilities": probabilities.tolist(),
            }
            with open(record_path, "a") as file:
                file.write(json.dumps(record) + "\n")

    totals = torch.tensor([trigger_sum, timeval_sum, half_count], dtype=torch.float64, device=device)
    dist.all_reduce(totals)
    if dist.get_rank() == 0:
        result = (
            f"TriggerAcc: {(totals[0] / totals[2]).item():.6f}\n"
            f"TimVal:     {(totals[1] / totals[2]).item():.6f}\n"
            f"TriggerSum: {totals[0].item():.9f}\n"
            f"TimValSum:  {totals[1].item():.9f}\n"
            f"Count:      {int(totals[2].item())}\n"
        )
        print(result, end="")
        if args.output:
            with open(args.output, "w") as file:
                file.write(result)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
