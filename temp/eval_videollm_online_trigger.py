import argparse
import json
import math
import os
import sys
import time

import decord
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F


SYSTEM_PROMPT = (
    "A multimodal AI assistant is helping users with some activities. "
    "Below is their conversation, interleaved with the list of video frames received by the assistant."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/root/videollm-online")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter", default="chenjoya/videollm-online-8b-v1plus")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.725)
    parser.add_argument("--tolerance_frames", type=int, default=2)
    parser.add_argument("--vision_batch_size", type=int, default=64)
    parser.add_argument("--context_seconds", type=int, default=600)
    parser.add_argument("--max_halves", type=int, default=-1)
    parser.add_argument("--max_seconds", type=int, default=-1)
    parser.add_argument("--output_dir", required=True)
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


def load_frames(reader, source_fps, end_second, fps):
    seconds = torch.arange(0, end_second + 1 / fps, 1 / fps).tolist()
    indices = [min(len(reader) - 1, round(second * source_fps)) for second in seconds]
    arrays = reader.get_batch(indices).asnumpy()
    frames = torch.from_numpy(arrays).permute(0, 3, 1, 2).float()
    height, width = frames.shape[-2:]
    scale = 384 / max(height, width)
    resized_h = max(2, round(height * scale / 2) * 2)
    resized_w = max(2, round(width * scale / 2) * 2)
    frames = F.interpolate(frames, size=(resized_h, resized_w), mode="bicubic", align_corners=False)
    pad_h = 384 - resized_h
    pad_w = 384 - resized_w
    frames = F.pad(frames, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    return frames.clamp(0, 255)


def reset_state(tokenizer, device):
    start_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}],
        add_stream_prompt=True,
        return_tensors="pt",
    ).to(device)
    return start_ids, None


@torch.inference_mode()
def evaluate_half(model, tokenizer, frames, fps, threshold, context_seconds, vision_batch_size):
    device = next(model.parameters()).device
    frame_embeddings = []
    for batch in frames.split(vision_batch_size):
        frame_embeddings.append(model.visual_embed(batch.to(device)).view(len(batch), -1, model.config.hidden_size))
    frame_embeddings = torch.cat(frame_embeddings)

    interval_id = model.config.frame_token_interval_id
    eos_id = model.config.eos_token_id
    added_stream_prompt = tokenizer.apply_chat_template(
        [{}], add_stream_prompt=True, return_tensors="pt"
    ).to(device)
    added_generation_prompt = tokenizer.apply_chat_template(
        [{}], add_stream_generation_prompt=True, return_tensors="pt"
    ).to(device)
    output_buffer = torch.zeros(1, 100, device=device, dtype=torch.long)
    from models import fast_greedy_generate

    predictions = []
    probabilities = []
    last_ids, past_key_values = reset_state(tokenizer, device)
    reset_frames = max(1, context_seconds * fps)
    for index, frame_embed in enumerate(frame_embeddings):
        if index and index % reset_frames == 0:
            last_ids, past_key_values = reset_state(tokenizer, device)
        elif past_key_values is not None and last_ids.item() == eos_id:
            last_ids = torch.cat([last_ids, added_stream_prompt], dim=1)

        inputs_embeds = torch.cat(
            [model.get_input_embeddings()(last_ids).view(1, -1, model.config.hidden_size), frame_embed[None]],
            dim=1,
        )
        outputs = model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        scores = outputs.logits[:, -1].float().softmax(dim=-1)
        p_continue = scores[0, interval_id]
        probability = 1 - p_continue
        prediction = int(p_continue < threshold)
        predictions.append(prediction)
        probabilities.append(float(probability))

        if prediction:
            next_scores = scores.clone()
            next_scores[0, interval_id] = 0
            last_ids = next_scores.argmax(dim=-1, keepdim=True)
            generation_embeds = model.get_input_embeddings()(added_generation_prompt)
            output_ids, past_key_values = fast_greedy_generate(
                model=model,
                inputs_embeds=generation_embeds,
                past_key_values=past_key_values,
                eos_token_id=eos_id,
                inplace_output_ids=output_buffer,
            )
            last_ids = output_ids[:, -1:]
        else:
            last_ids = torch.tensor([[interval_id]], device=device)
    return predictions, probabilities


def main():
    args = parse_args()
    sys.path.insert(0, args.repo)
    from models.live_llama import build_live_llama

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model, tokenizer = build_live_llama(
        is_training=False,
        llm_pretrained="meta-llama/Meta-Llama-3-8B-Instruct",
        vision_pretrained="google/siglip-large-patch16-384",
        finetune_modules=["connector"],
        lora_modules="model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$",
        lora_r=128,
        lora_alpha=256,
        set_vision_inside=True,
        resume_from_checkpoint=args.adapter,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        frame_resolution=384,
        frame_token_cls=True,
        frame_token_pooled=[3, 3],
        frame_num_tokens=10,
        frame_token_interval=",",
        max_num_frames=1200,
    )
    model.to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    record_path = os.path.join(args.output_dir, f"records_{dist.get_rank():02d}.jsonl")
    frame = pd.read_parquet(args.manifest)
    if args.max_halves >= 0:
        frame = frame.iloc[: args.max_halves]
    started = time.time()
    completed = 0
    for half_index, row in frame.iterrows():
        if half_index % dist.get_world_size() != dist.get_rank():
            continue
        events = event_times(row)
        if not events:
            continue
        reader = decord.VideoReader(row.video_path, ctx=decord.cpu(0), num_threads=1)
        source_fps = float(reader.get_avg_fps())
        end_second = min(events[-1], math.floor((len(reader) - 1) / source_fps))
        if args.max_seconds >= 0:
            end_second = min(end_second, args.max_seconds)
        frames = load_frames(reader, source_fps, end_second, args.fps)
        labels = [0] * len(frames)
        for event in events:
            index = round(event * args.fps)
            if index < len(labels):
                labels[index] = 1
        predictions, probabilities = evaluate_half(
            model, tokenizer, frames, args.fps, args.threshold, args.context_seconds, args.vision_batch_size
        )
        record = {
            "half": int(half_index),
            "tolerance": args.tolerance_frames,
            "labels": labels,
            "predictions": predictions,
            "response_probabilities": probabilities,
        }
        with open(record_path, "a") as file:
            file.write(json.dumps(record) + "\n")
        completed += 1
        print(
            json.dumps({"rank": dist.get_rank(), "half": int(half_index), "done": completed,
                        "positions": len(labels), "elapsed": time.time() - started}),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
