# ov_train/train.py
"""Two-stage trainer for OneVision-based StreamMind.

Stage 1: train base CausalLM on caption generation; gate frozen.
Stage 2: freeze base, train only the gate head on silence/response labels.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch
from transformers import AutoProcessor, HfArgumentParser, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ov_train.onevision_stream import StreamOneVision
from ov_train.soccer_dataset import OneVisionCollator, SoccerOneVisionDataset


@dataclass
class ModelArgs:
    model_name_or_path: str = field(
        default="/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct"
    )


@dataclass
class DataArgs:
    split: str = "train"
    video_backend: str = "frames"  # "frames" or "codec"
    num_frames: int = 16
    window_seconds: float = 8.0
    max_samples: int | None = None


@dataclass
class StageArgs:
    stage: int = 1  # 1 = train LLM, 2 = train gate


def apply_stage_freeze(model: StreamOneVision, stage: int):
    if stage == 1:
        for p in model.gate.parameters():
            p.requires_grad = False
    elif stage == 2:
        for p in model.base.parameters():
            p.requires_grad = False
        for p in model.gate.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown stage {stage}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[stage {stage}] trainable {trainable/1e6:.1f}M / total {total/1e9:.2f}B")


def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, StageArgs, TrainingArguments))
    model_args, data_args, stage_args, training_args = parser.parse_args_into_dataclasses()

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    model = StreamOneVision(model_args.model_name_or_path)
    apply_stage_freeze(model, stage_args.stage)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    train_ds = SoccerOneVisionDataset(
        processor,
        split=data_args.split,
        video_backend=data_args.video_backend,
        num_frames=data_args.num_frames,
        window_seconds=data_args.window_seconds,
        max_samples=data_args.max_samples,
        stage=stage_args.stage,
    )
    collator = OneVisionCollator(pad_token_id=processor.tokenizer.pad_token_id or 0)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
