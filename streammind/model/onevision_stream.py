# streammind/model/onevision_stream.py
"""StreamMind-style wrapper around LLaVA-OneVision-2 for two-stage training.

Stage 1: train the underlying CausalLM on video captions (gate frozen).
Stage 2: freeze base, train only the gate head on silence/response labels.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class StreamOneVision(nn.Module):
    def __init__(self, model_name_or_path: str, gate_class_weights=(0.15, 0.85)):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        hidden = self.base.config.text_config.hidden_size  # 4096 for Qwen3-8B
        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        # gate stays in fp32 for numerical stability; small enough not to matter
        self.register_buffer(
            "_gate_w",
            torch.tensor(gate_class_weights, dtype=torch.float32),
            persistent=False,
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.base.gradient_checkpointing_enable(**kwargs)

    def forward(
        self,
        gate_label: torch.LongTensor | None = None,
        gate_pos: torch.LongTensor | None = None,
        **inputs,
    ):
        need_h = gate_label is not None
        out = self.base(**inputs, output_hidden_states=need_h, return_dict=True)
        loss = out.loss

        if need_h:
            h = out.hidden_states[-1]                              # [B, T, H]
            b_idx = torch.arange(h.size(0), device=h.device)
            g_logits = self.gate(h[b_idx, gate_pos].float())       # [B, 2]
            gate_loss = F.cross_entropy(
                g_logits, gate_label, weight=self._gate_w.to(g_logits.device)
            )
            loss = gate_loss if loss is None else loss + gate_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=out.logits,
            past_key_values=getattr(out, "past_key_values", None),
            hidden_states=out.hidden_states if need_h else None,
        )
