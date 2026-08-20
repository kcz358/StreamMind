"""Evaluate JoyAI-VL-Interaction on MatchTime using its official agent system.

Protocol mirrors services/webinfer/live_adapter.py and memory_summarizer.py from the JD
reference release: one user turn per second carrying a single frame, the standing query
injected on the first user turn of each chunk, the model's own action fed back as history,
and the short-term window reset every ``--chunk-turns`` turns. With ``--memory`` the
3-tier summary memory is enabled: each finished chunk is summarised by a separate
summariser model, and every ``--compress-every`` mid-term summaries are compressed into
long-term memory. Both tiers are injected as a Video History prefix.
"""

import argparse
import base64
import io
import json
import math
import os
import time

import decord
import pandas as pd
import requests
from PIL import Image

SYSTEM_PROMPT = """You are a real-time video streaming assistant observing a continuous camera feed frame by frame. The last frame represents the current moment.
## Action Format
At every inference step you MUST choose exactly one of the following two actions:
**Stay silent** — output ONLY:
</silence>
Choose this when nothing noteworthy has changed in the scene, no user query is pending, or there is nothing useful to say.
**Speak** — output the token followed by a concise reply:
</response> Your reply here.
Choose this when you observe something worth reporting or a significant state change, or when you can answer a user question based on available evidence.

## Important
There is NO delegation action. NEVER output </delegation> or hand off questions to any background solver."""
QUERY_HEADER = "[User Query (IMPORTANT — follow this instruction)]"
VIDEO_HISTORY_HEADER = (
    "[Video History]\n"
    "The following are summaries of earlier video segments you can no longer see. "
    "Use them as background context, but always prioritize the current visual frames "
    "and the User Query below when making decisions.\n"
    "IMPORTANT: These summaries are written by an external system in a descriptive style. "
    "Do NOT imitate their writing style in your responses.\n"
)
QUERY = "Watch this soccer match and alert me whenever a noteworthy event occurs."
SILENCE = "</silence>"
RESPONSE = "</response>"

DETAILED_SUMMARY_PROMPT = """You are writing mid-term memory for a long-running video agent. The user message contains timestamped key frames for Chunk {chunk_index}, covering {frame_range}; each image is preceded by its sampled timestamp span. These frames will be unavailable afterward, so your paragraph must preserve the key evidence that downstream models need for recall, reasoning, and answering follow-up questions once the visuals are gone.

[Output Format]
- Write a SINGLE factual paragraph. Use the full output budget when the chunk is information-dense; for near-static or nearly duplicate frames, 1-2 brief sentences suffice — do not pad for length.
- When identity, text, or numeric values are uncertain, do not guess — mark them as "possibly X", "approx. X", or "unreadable"; even when unreadable, preserve the observable category, appearance, action, or positional features.

[Information to Preserve (by priority, highest first)]
- End-of-chunk handoff state, irreversible events and state changes, intermediate results and task progress, and causal chains.
- Readable text, labels, numbers, counts, identifiers, first-time or anomalous events, and final positions of important objects.
- Inventory of entities present, scene layout, and sparse time anchors for important events.

[Writing Rules]
- Use at most 3-5 explicit time anchors, preferring merged ranges of roughly {preferred_time_span}.
- Use only details directly visible in the provided key frames; describe in temporal order.
- Avoid generic wrap-up phrases and meta commentary.

Output ONLY the paragraph."""

BATCH_COMPRESS_PROMPT = """You are compressing multiple mid-term memory segments into long-term memory for a long-running video agent. The original frames for period {merged_range} will no longer be available, so the output must preserve the key evidence that downstream models need for recall, reasoning, and answering follow-up questions afterward.

MID-TERM SUMMARIES TO MERGE:
{summaries_text}

[Task Definition]
This is a compression task, not a simple merge. Prioritize and discard to reduce information density. Only the final segment's end state matters downstream; earlier end states superseded by later events are process information.

[Output Format]
- Write a SINGLE unified factual paragraph (NOT separate per-segment summaries).
- Carry through uncertainty markers from the mid-term summaries.

[Retention Priority]
- Must preserve: the final handoff state, cross-segment irreversible events, readable text and "item → value" mappings, and causal chains.
- Should preserve: entities persisting across segments, overall progress structure, first-time or anomalous events, and final positions.
- Compressible: intermediate states overwritten later, unchanged background, and repeated similar actions.

[Time References]
- Use 3-7 explicit time anchors across the paragraph, merging continuous or repeated neighbouring actions into one range.

Output ONLY the unified narrative text, nothing else."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", default="joyai")
    parser.add_argument("--chunk_turns", type=int, default=100)
    parser.add_argument("--max_pixels", type=int, default=262144)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_halves", type=int, default=-1)
    parser.add_argument("--max_seconds", type=int, default=-1)
    parser.add_argument("--tolerance_seconds", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--summarizer_base_url", default="")
    parser.add_argument("--summarizer_model", default="summarizer")
    parser.add_argument("--summarizer_frames", type=int, default=10)
    parser.add_argument("--summarizer_max_pixels", type=int, default=262144)
    parser.add_argument("--compress_every", type=int, default=5)
    parser.add_argument("--mid_term_max_tokens", type=int, default=4000)
    parser.add_argument("--long_term_max_tokens", type=int, default=2000)
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


def encode_frame(reader, fps, second, max_pixels):
    index = min(len(reader) - 1, max(0, round(second * fps)))
    image = Image.fromarray(reader[index].asnumpy()).convert("RGB")
    pixels = image.width * image.height
    if max_pixels > 0 and pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def user_turn(second, data_url, prefix):
    content = []
    if prefix:
        content.append({"type": "text", "text": prefix})
    content.append({"type": "text", "text": f"<{second:.1f} seconds>"})
    content.append({"type": "image_url", "image_url": {"url": data_url}})
    return {"role": "user", "content": content}


def action_probability(logprobs):
    """Probability mass the first generated token puts on </response> vs </silence>."""
    scores = {}
    for entry in logprobs:
        token = entry.get("token", "")
        for marker in (SILENCE, RESPONSE):
            if marker in token and marker not in scores:
                scores[marker] = math.exp(entry["logprob"])
    silence = scores.get(SILENCE, 0.0)
    response = scores.get(RESPONSE, 0.0)
    total = silence + response
    return response / total if total > 0 else 0.0


class Memory:
    """3-tier summary memory backed by a separate summariser model."""

    def __init__(self, session, args):
        self.session = session
        self.args = args
        self.reset()

    def reset(self):
        self.mid_term = []
        self.long_term = ""

    def prefix(self):
        parts = []
        if self.long_term:
            parts.append(self.long_term)
        for entry in self.mid_term:
            parts.append(f"<{entry['frame_range']}>\n{entry['summary_text']}")
        if not parts:
            return ""
        return VIDEO_HISTORY_HEADER + "\n\n".join(parts)

    def _chat(self, content, max_tokens):
        payload = {
            "model": self.args.summarizer_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        reply = self.session.post(
            f"{self.args.summarizer_base_url}/chat/completions", json=payload, timeout=900
        ).json()
        return (reply["choices"][0]["message"]["content"] or "").strip()

    def add_chunk(self, chunk_index, frames):
        """frames: list of (second, data_url) covering the finished chunk."""
        if not frames:
            return
        stride = max(1, len(frames) // self.args.summarizer_frames)
        key_frames = frames[::stride]
        frame_range = f"{frames[0][0]:.1f} seconds ~ {frames[-1][0]:.1f} seconds"
        content = [
            {
                "type": "text",
                "text": DETAILED_SUMMARY_PROMPT.format(
                    chunk_index=chunk_index,
                    frame_range=frame_range,
                    preferred_time_span="10 seconds",
                ),
            }
        ]
        for second, data_url in key_frames:
            content.append({"type": "text", "text": f"<{second:.1f} seconds>"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        summary = self._chat(content, self.args.mid_term_max_tokens)
        self.mid_term.append({"frame_range": frame_range, "summary_text": summary})
        if len(self.mid_term) >= self.args.compress_every:
            self.compress()

    def compress(self):
        summaries_text = "\n\n".join(
            f"<{entry['frame_range']}>\n{entry['summary_text']}" for entry in self.mid_term
        )
        merged_range = f"{self.mid_term[0]['frame_range']}-{self.mid_term[-1]['frame_range']}"
        content = [
            {
                "type": "text",
                "text": BATCH_COMPRESS_PROMPT.format(
                    summaries_text=summaries_text, merged_range=merged_range
                ),
            }
        ]
        compressed = self._chat(content, self.args.long_term_max_tokens)
        self.long_term = (self.long_term + "\n\n" + compressed).strip()
        self.mid_term = []


def run_half(session, args, reader, fps, seconds, memory):
    predictions = []
    probabilities = []
    messages = []
    chunk_frames = []
    chunk_index = 0
    if memory is not None:
        memory.reset()
    for turn, second in enumerate(seconds):
        if turn and turn % args.chunk_turns == 0:
            if memory is not None:
                memory.add_chunk(chunk_index, chunk_frames)
            messages = []
            chunk_frames = []
            chunk_index += 1
        data_url = encode_frame(reader, fps, second, args.max_pixels)
        chunk_frames.append((second, data_url))
        if messages:
            prefix = ""
        else:
            history = memory.prefix() if memory is not None else ""
            prefix = "\n\n".join(part for part in (history, f"{QUERY_HEADER}\n{QUERY}") if part)
        messages.append(user_turn(second, data_url, prefix))
        payload = {
            "model": args.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
        }
        reply = session.post(
            f"{args.base_url}/chat/completions", json=payload, timeout=600
        ).json()["choices"][0]
        text = (reply["message"]["content"] or "").strip()
        speaks = text.startswith(RESPONSE)
        predictions.append(int(speaks))
        probabilities.append(action_probability(reply["logprobs"]["content"][0]["top_logprobs"]))
        messages.append({"role": "assistant", "content": text if text else SILENCE})
    return predictions, probabilities


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    record_path = os.path.join(args.output_dir, f"records_{args.rank:02d}.jsonl")
    session = requests.Session()
    memory = Memory(session, args) if args.memory else None

    frame = pd.read_parquet(args.manifest)
    if args.max_halves >= 0:
        frame = frame.iloc[: args.max_halves]
    started = time.time()
    done = 0

    for half_index, row in frame.iterrows():
        if half_index % args.world_size != args.rank:
            continue
        events = event_times(row)
        if not events:
            continue
        reader = decord.VideoReader(row.video_path, ctx=decord.cpu(0), num_threads=1)
        fps = float(reader.get_avg_fps())
        end = min(events[-1], math.floor((len(reader) - 1) / fps))
        if args.max_seconds >= 0:
            end = min(end, args.max_seconds)
        seconds = list(range(end + 1))
        labels = [0] * len(seconds)
        for event in events:
            if event <= end:
                labels[event] = 1
        predictions, probabilities = run_half(session, args, reader, fps, seconds, memory)
        record = {
            "half": int(half_index),
            "tolerance": args.tolerance_seconds,
            "labels": labels,
            "predictions": predictions,
            "response_probabilities": probabilities,
        }
        with open(record_path, "a") as file:
            file.write(json.dumps(record) + "\n")
        done += 1
        print(
            json.dumps(
                {
                    "rank": args.rank,
                    "half": int(half_index),
                    "done": done,
                    "positions": len(seconds),
                    "elapsed": time.time() - started,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
