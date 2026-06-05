"""Train entry for OneVision-2 + StreamMind paper-faithful pipeline.

Drop-in replacement for ``streammind.train_new_stream``: uses the same
``DataCollatorForstreamDataset`` + ``StreamMindTrainer``, but the model is
``OneVisionStreamForCausalLM`` (OV2 backbone + Mamba EPFE + Qwen3 ClsNet).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch
from transformers import AutoProcessor, AutoTokenizer, HfArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ported verbatim from streammind/train_new_stream.py:564-587 and 105-130 to
# avoid circular-import issues when ``streammind`` is half-loaded.
import transformers
from typing import Optional, Sequence, Dict, Union, Any


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(default=512)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = field(default=False)
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


@dataclass
class DataCollatorForstreamDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        instance = instances[0]
        batch = dict()
        batch["timestamp"] = instance["timestamp"]
        batch["labels"] = instance["labels"]
        batch["input_ids"] = instance["input_ids"]
        batch["caption_info"] = instance["caption_info"]
        batch["video_path"] = instance["video_path"]
        if "image" in instance.keys():
            batch["images"] = [instance["image"], ["image"]]
        else:
            batch["images"] = [instance["video"], ["video"]]
        batch["attention_mask"] = None
        batch["past_review_caption"] = instance["past_review_caption"]
        batch["data_type"] = instance["data_type"]
        batch["model_type"] = instance["model_type"]
        return batch


from streammind.streammind_trainer_score import StreamMindTrainer

from ov_train.datasets import LazySupervisedDataset, DataArguments
from ov_train.onevision_stream import OneVisionStreamForCausalLM


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/data/kaichen/LLaVA-OneVision-1.5-RL/pretrained/LLaVA-OneVision-2-8B-Instruct"
    )
    mm_projector_type: str = field(default="mamba")
    freeze_backbone: bool = field(default=False)
    soccer_dataset_train_llm: bool = field(default=False)
    soccer_dataset_train_cls: bool = field(default=False)


def apply_freeze(model: OneVisionStreamForCausalLM, model_args: ModelArguments) -> None:
    if model_args.soccer_dataset_train_cls:
        model.requires_grad_(False)
        for name, param in model.get_model().mm_projector.named_parameters():
            if "cls" in name:
                param.requires_grad = True
    elif model_args.soccer_dataset_train_llm:
        for name, param in model.get_model().mm_projector.named_parameters():
            if "cls" in name:
                param.requires_grad = False
        if model_args.freeze_backbone:
            model.get_model().language_model.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[freeze] trainable {trainable/1e6:.1f}M / total {total/1e9:.2f}B", flush=True)


def _resolve_image_processor(processor):
    """OneVision's ``AutoProcessor`` exposes its image processor differently
    across releases; pick the first attribute that carries ``image_mean``."""
    for attr in ("image_processor", "video_processor"):
        candidate = getattr(processor, attr, None)
        if candidate is not None and hasattr(candidate, "image_mean"):
            return candidate
    if hasattr(processor, "image_mean"):
        return processor
    raise AttributeError(
        "Could not locate an image_processor with image_mean on the OneVision processor."
    )


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    image_processor = _resolve_image_processor(processor)
    data_args.video_processor = image_processor
    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    data_args.soccer_dataset_train_llm = model_args.soccer_dataset_train_llm

    model = OneVisionStreamForCausalLM(model_args.model_name_or_path)
    model.add_streammind_special_tokens(tokenizer)
    apply_freeze(model, model_args)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path or "",
        tokenizer=tokenizer,
        data_args=data_args,
    )
    collator = DataCollatorForstreamDataset(tokenizer=tokenizer)

    trainer = StreamMindTrainer(
        data_args,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
