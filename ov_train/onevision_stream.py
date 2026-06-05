"""OneVision-2 wrapper inheriting StreamMind's splice/cls/EPFE logic.

Builds a Causal LM whose backbone is the LLaVA-OneVision-2 model
(``model.visual`` + ``model.language_model``) and whose multimodal
projector is the Qwen3-based ``Video_Mamba_seq`` from ``ov_train.projector``
(EPFE Mamba + Qwen3 ClsNet).
We deliberately avoid the original LLaVA-style ``initialize_vision_modules``
plumbing — OV2 ships its own vision tower and we wrap it directly.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ov_train.onevision_arch import OneVisionStreamMetaForCausalLM
from ov_train.projector import Video_Mamba_seq


class _ProjectorConfig:
    """Minimal config object satisfying ``Video_Mamba_seq.__init__``."""

    def __init__(self, mm_hidden_size: int, hidden_size: int, mm_projector_type: str = "mamba"):
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size
        self.mm_projector_type = mm_projector_type


class _InnerModel(nn.Module):
    """Holds the full OV2 model + StreamMind's Mamba projector.

    ``OneVisionStreamMetaForCausalLM`` expects::

        self.get_model() -> this object
        this.get_vision_tower() -> the vision tower
        this.mm_projector       -> the Mamba+ClsNet projector
        this.embed_tokens(ids)  -> token embeddings (delegates to Qwen3)

    We keep the full ``ov_model`` as a single submodule and expose its
    ``visual`` / ``language_model`` / ``lm_head`` via properties so that
    ``resize_token_embeddings`` (which replaces ``lm_head`` with a new
    ``Linear``) never leaves us with stale references.
    """

    def __init__(self, ov_model, mm_projector: nn.Module):
        super().__init__()
        self.ov_model = ov_model
        self.mm_projector = mm_projector

    @property
    def language_model(self):
        return self.ov_model.model.language_model

    @property
    def lm_head(self):
        return self.ov_model.lm_head

    def get_vision_tower(self):
        return self.ov_model.model.visual

    def embed_tokens(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings()(input_ids)


class OneVisionStreamForCausalLM(nn.Module, OneVisionStreamMetaForCausalLM):
    """OV2 backbone + StreamMind splice / cls / EPFE logic.

    Not a subclass of ``Qwen3ForCausalLM`` — we delegate to OV2's already-
    instantiated ``language_model`` and ``lm_head`` directly. This matches the
    ``Videollama2MistralForCausalLM`` pattern (splice multimodal embeddings,
    then run the LLM trunk, then ``lm_head``).
    """

    def __init__(
        self,
        model_name_or_path: str,
        mm_hidden_size: Optional[int] = None,
        projector_hidden_size: Optional[int] = None,
        mm_projector_type: str = "mamba",
        sample_per: float = 0.5,
        sample_type: str = "all",
        attn_implementation: str = "flash_attention_2",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        ov_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        self.config = ov_model.config

        if mm_hidden_size is None:
            # OV2's vision tower projects each merged patch to ``out_hidden_size``
            # (4096 for OV2 8B); that is what feeds the Mamba EPFE.
            mm_hidden_size = self.config.vision_config.out_hidden_size
        if projector_hidden_size is None:
            projector_hidden_size = self.config.text_config.hidden_size

        proj_cfg = _ProjectorConfig(
            mm_hidden_size=mm_hidden_size,
            hidden_size=projector_hidden_size,
            mm_projector_type=mm_projector_type,
        )
        mm_projector = Video_Mamba_seq(proj_cfg)
        mm_projector.to(dtype=dtype)

        # ``OneVisionStreamMetaForCausalLM.temporal_aggregator`` dispatches on
        # ``self.config.mm_projector_type``; OV2's own config has no such field.
        self.config.mm_projector_type = mm_projector_type
        self.config.mm_hidden_size = mm_hidden_size

        self.model = _InnerModel(ov_model, mm_projector)
        self.vocab_size = self.config.text_config.vocab_size

        # Freeze the vision tower by default (matches the original StreamMind /
        # VideoLLaMA2 design where CLIP is always frozen). EPFE helper relies on
        # this to skip activation storage during forward.
        for p in self.model.get_vision_tower().parameters():
            p.requires_grad = False

        self.train_iteration = 0
        self.frame_feature = None
        self.past_review_caption = None
        self.past_review_caption_list: List = []
        self.interval_id_list: List = []
        self.time_list: List = []
        self.sample_per = sample_per
        self.sample_type = sample_type
        self.loss_fct = CrossEntropyLoss()

    def get_model(self) -> _InnerModel:
        return self.model

    @property
    def lm_head(self) -> nn.Linear:
        return self.model.ov_model.lm_head

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """Resize OV2's embedding + ``lm_head`` to fit ``new_num_tokens``.

        Only grows, never truncates: OV2 ships with embeddings padded to 151936
        while the tokenizer only has 151665 in-use slots, so requesting a small
        ``new_num_tokens`` would otherwise discard valid weights.
        """
        cur = self.model.ov_model.get_input_embeddings().weight.shape[0]
        target = max(cur, new_num_tokens)
        if target != cur:
            self.model.ov_model.resize_token_embeddings(target)
            self.vocab_size = target
            self.config.text_config.vocab_size = target
        return self.model.ov_model.get_input_embeddings()

    def add_streammind_special_tokens(self, tokenizer) -> Tuple[int, int]:
        """Register StreamMind's ``</silence>`` / ``</response>`` tokens.

        Adds the two markers to ``tokenizer`` (as additional special tokens),
        resizes the OV2 embedding / ``lm_head`` to cover them, and stores
        the resulting ids on ``self.model.mm_projector`` so the Qwen3 ClsNet
        slices ``prompt_time_lable`` by the correct ids instead of the
        Mistral-era 32000/32001 defaults. Returns ``(silence_id, response_id)``.
        """
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["</silence>", "</response>"]}
        )
        self.resize_token_embeddings(len(tokenizer))
        silence_id = tokenizer.convert_tokens_to_ids("</silence>")
        response_id = tokenizer.convert_tokens_to_ids("</response>")
        self.model.mm_projector.silence_token_id = silence_id
        self.model.mm_projector.response_token_id = response_id
        return silence_id, response_id

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.language_model.gradient_checkpointing_enable(**kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cls_output=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Mirror ``Videollama2MistralForCausalLM.forward``.

        * ``timestamp`` in ``kwargs`` -> StreamMind streaming path (splice + cls).
        * Otherwise -> standard ``prepare_inputs_labels_for_multimodal`` path.
        * If the splice returns ``cls_output`` (stage-2 cls mode), return it as-is.
        """
        if inputs_embeds is None:
            if "timestamp" in kwargs:
                (
                    input_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    cls_output,
                ) = self.prepare_inputs_labels_for_multimodal_score_stream(
                    input_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    sample_per=self.sample_per,
                    sample_type=self.sample_type,
                    **kwargs,
                )
            else:
                (
                    input_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                )

        if cls_output is not None:
            return cls_output

        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        llm_output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

        if kwargs.pop("llm_eval", None):
            return llm_output, labels
        return llm_output
