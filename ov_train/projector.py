"""Mamba EPFE + Qwen3-based ClsNet projector (Shallow Layer Transfer with Qwen3 backbone).

Ports ``streammind.model.multimodal_projector.builder.Video_Mamba_seq`` to use a
Qwen3 mini-LLM as the cognition gate (ClsNet) instead of Mistral, so that the
gate's tokenizer / vocabulary / embedding space match a Qwen3-based main LLM
(e.g. LLaVA-OneVision-2 8B). Only the gate model and two label-marker token ids
change; the EPFE backbone (``PreNet`` -> ``VideoMamba`` -> ``PostNet``) and the
per-frame autoregressive sequence layout are reused verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import Qwen3Config
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3 import Qwen3ForCausalLM

from streammind.constants import IGNORE_INDEX
from streammind.model.multimodal_projector.builder import PreNet, PostNet
from streammind.model.multimodal_projector.ssm import VideoMamba


@dataclass
class SSMConfig:
    """Mirror of streammind's SSMConfig (re-declared so callers don't need to import it)."""
    d_code = 1024
    d_model = 2048
    n_ssm = 1
    n_classes = 400
    lr = 1.4e-4
    lr_min = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0.02
    scheduler = "plateau"


class Qwen3ForCausalLM_cls(Qwen3ForCausalLM):
    """Tiny Qwen3 CausalLM with the same weighted-CE loss as the original Mistral
    ``MistralForCausalLM_cls``: a uniform weight of 1.0 for ids ``[0, vocab-2)``
    plus ``[0.15, 0.85]`` for the last two ids (silence / response in the gate's
    own ``vocab_size=2`` mini-vocab).
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Match the original Mistral gate's default: return_dict=True when unset.
        if return_dict is None:
            return_dict = True
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            weight_list = [1.0] * (self.config.vocab_size - 2) + [0.15, 0.85]
            loss_fct = CrossEntropyLoss(
                weight=torch.tensor(weight_list, device=shift_logits.device)
            )
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            return (loss, logits, outputs.past_key_values, None, None)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=None,
        )


class ClsNet(nn.Module):
    """4-layer Qwen3-based cognition gate (Shallow Layer Transfer with ``vocab_size=2``).

    The hidden size, head count, MLP intermediate size, and RMSNorm eps mirror
    Qwen3-8B so that an upstream ``embed_tokens`` lookup on the main LLM's
    vocabulary produces tensors the gate can consume directly (4096-dim).
    """

    def __init__(self, hidden_size: int = 4096, num_layers: int = 4):
        super().__init__()
        cfg = Qwen3Config(
            vocab_size=2,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=12288,
            head_dim=128,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            attention_bias=False,
        )
        self.cls_model = Qwen3ForCausalLM_cls(cfg)

    def forward(self, x, cls_labels, cls_attention_mask):
        return self.cls_model(
            inputs_embeds=x,
            labels=cls_labels,
            attention_mask=cls_attention_mask,
        )


class Video_Mamba_seq(nn.Module):
    """Faithful port of ``streammind.model.multimodal_projector.builder.Video_Mamba_seq``
    with the Mistral-based ClsNet swapped for our Qwen3-based ``ClsNet`` and the
    two hardcoded silence/response token ids made configurable.

    Args:
        model_config: object with ``.mm_hidden_size`` and ``.hidden_size``.
        silence_token_id: main-LLM token id used as the silence label marker in
            ``prompt_time_lable`` (replaces the original literal ``32000``).
        response_token_id: main-LLM token id used as the response label marker
            (replaces the original literal ``32001``).
    """

    def __init__(
        self,
        model_config,
        silence_token_id: int = 32000,
        response_token_id: int = 32001,
    ):
        super().__init__()
        self.pre_net = PreNet(model_config.mm_hidden_size, model_config.hidden_size)
        mamba_config = SSMConfig()
        mamba_config.d_code = model_config.hidden_size
        mamba_config.d_model = model_config.hidden_size
        self.mamba_model = VideoMamba(mamba_config)
        self.post_net = PostNet(model_config.hidden_size, model_config.hidden_size)
        self.cls_net = ClsNet(hidden_size=model_config.hidden_size, num_layers=4)
        self.silence_token_id = silence_token_id
        self.response_token_id = response_token_id
        self.time_list = []
        self.videoid = 0

    def forward(self, x, cls_inference = False,cls_training = False,cls_demo = False,frames_features_shape = [],prompt_time_input_ids = None,prompt_time_lable = None):
        b, t, l, d = x.shape
        x = torch.mean(x, dim=2)
        x = einops.rearrange(x, "b t d -> (b t) d", b=b, t=t)
        # import pdb
        # pdb.set_trace()
        x = self.pre_net(x)
        x = einops.rearrange(x, "(b t) d -> b t d", b=b, t=t)
        x = self.mamba_model(x)
        x = einops.rearrange(x, "b t d -> (b t) d")
        x = self.post_net(x)
        x = einops.rearrange(x, "(b t) d -> b t d", b=b, t=t)

        if cls_training or cls_inference:
            if prompt_time_input_ids is not None and prompt_time_input_ids.numel()>1:
                pad_token_id = 0
                input_embeds = []
                cls_labels = []
                X_prompt_indices = torch.where(prompt_time_input_ids == -201)[1]
                X_prediction_indices = torch.where(prompt_time_lable == self.silence_token_id)[1] #找到</silence>这个token的位置，</response>是self.response_token_id
                #[INST] <<SYS>>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n<</SYS>>\n\nWhat are the prerequisites for the next task?<video>\n [/INST] </silence> </s>"
                prompt_template_inputs = self.cls_net.cls_model.model.embed_tokens(prompt_time_input_ids[: , : X_prompt_indices])#这个是sys那部分
                prompt_template_inputs_requirements = self.cls_net.cls_model.model.embed_tokens(prompt_time_input_ids[ : , X_prompt_indices + 1 : X_prediction_indices])#这个是用户需求
                prompt_template_inputs_rest = self.cls_net.cls_model.model.embed_tokens(prompt_time_input_ids[ : , X_prediction_indices + 1 :])

                prompt_template_labels = prompt_time_lable[:,:X_prompt_indices].to(x.device)
                prompt_template_labels_requirements = prompt_time_lable[:,X_prompt_indices + 1 : X_prediction_indices].to(x.device)
                prompt_template_labels_rest = prompt_time_lable[:,X_prediction_indices + 1:].to(x.device)

                eos_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([0]).to(x.device)).unsqueeze(0)
                caption_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([1]).to(x.device)).unsqueeze(0)
                input_embeds = []
                cls_labels = []
                start_feature_idx = [0] + frames_features_shape[:-1]
                for idx, end_frame_idx in enumerate(frames_features_shape):
                    cur_frame_feature = x[0][start_feature_idx[idx] : end_frame_idx]
                    if cur_frame_feature.shape[0] > 1:
                        input_embed = torch.cat([torch.cat([prompt_template_inputs,
                                                    frame.unsqueeze(0).unsqueeze(0),
                                                    prompt_template_inputs_requirements,
                                                    eos_target,
                                                    prompt_template_inputs_rest], dim=1) for frame in cur_frame_feature[:-1]])

                        cls_label= torch.cat([torch.cat([prompt_template_labels,
                                                    torch.full((1,1),IGNORE_INDEX).to(x.device),
                                                    prompt_template_labels_requirements,
                                                    torch.tensor([[self.silence_token_id]]).to(x.device),
                                                    prompt_template_labels_rest], dim=1) for _ in cur_frame_feature[:-1]])
                        input_embeds.append(input_embed)
                        cls_labels.append(cls_label)
                    input_embeds.append(torch.cat([prompt_template_inputs,
                                                    cur_frame_feature[-1].unsqueeze(0).unsqueeze(0),
                                                    prompt_template_inputs_requirements,
                                                    caption_target,
                                                    prompt_template_inputs_rest],dim=1))

                    cls_labels.append(torch.cat([prompt_template_labels,
                                                torch.full((1,1),IGNORE_INDEX).to(x.device),
                                                prompt_template_labels_requirements,
                                                torch.tensor([[self.response_token_id]]).to(x.device),
                                                prompt_template_labels_rest],dim = 1))

                input_embed = torch.cat(input_embeds)
                cls_label= torch.cat(cls_labels)
                # input_embed =  einops.rearrange(input_embeds, "(b t) c -> b t c", t=2)
                # cls_label =  einops.rearrange(cls_labels, "(b t)  -> b t ", t=2)
                if cls_label.shape[0]>4000:
                    cls_label = cls_label[:4000]
                    input_embed = input_embed[:4000]
                #all frame
                # input_embed = torch.nn.utils.rnn.pad_sequence(input_embeds,batch_first=True,padding_value=pad_token_id)
                # cls_label = torch.nn.utils.rnn.pad_sequence(cls_labels,batch_first=True,padding_value=IGNORE_INDEX)

                #mask
                # tf 5.x masking_utils expects 2-D attention_mask [B, T]; the
                # original code used element-wise .ne(pad) on the 3-D embed which
                # produced [B, T, D] and tf 4.x silently coerced. Collapse the
                # hidden dim with .any() to recover [B, T].
                cls_attention_mask = input_embed.ne(pad_token_id).any(dim=-1)

                if cls_training:
                    cls_output= self.cls_net(input_embed,cls_labels=cls_label,cls_attention_mask=cls_attention_mask)
                # cls_loss = self.cls_net(x,None)
                    return cls_output
                else:
                    cls_output= self.cls_net(input_embed,cls_labels=cls_label,cls_attention_mask=cls_attention_mask)

                    return cls_output,cls_label


            else:
                pad_token_id = 0
                input_embeds = []
                cls_labels = []
                start_feature_idx = [0] + frames_features_shape[:-1]
                for idx, end_frame_idx in enumerate(frames_features_shape):
                    cur_frame_feature = x[0][start_feature_idx[idx] : end_frame_idx]

                    eos_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([0]).to(x.device))
                    caption_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([1]).to(x.device))
                    ignore_tensor = torch.tensor([IGNORE_INDEX]).to(x.device)

                    if cur_frame_feature.shape[0] > 1:
                        input_embed = torch.cat([torch.cat([frame.unsqueeze(0),eos_target]) for frame in cur_frame_feature[:-1]])
                        eos_label = torch.cat([torch.cat([ignore_tensor, torch.tensor([0]).to(x.device)]) for _ in cur_frame_feature[:-1]])

                        input_embeds.append(torch.cat([input_embed, cur_frame_feature[-1].unsqueeze(0), caption_target]))
                        cls_labels.append(torch.cat([eos_label, ignore_tensor, torch.tensor([1]).to(x.device)]))
                    else:
                        input_embed = torch.cat([cur_frame_feature,caption_target])
                        caption_label = torch.cat([ignore_tensor, torch.tensor([1]).to(x.device)])
                        input_embeds.append(input_embed)
                        cls_labels.append(caption_label)
                input_embeds = torch.cat(input_embeds)
                cls_labels = torch.cat(cls_labels)
                input_embed =  einops.rearrange(input_embeds, "(b t) c -> b t c", t=2)
                cls_label =  einops.rearrange(cls_labels, "(b t)  -> b t ", t=2)
                if cls_label.shape[0]>4000:
                    cls_label = cls_label[:4000]
                    input_embed = input_embed[:4000]
                #all frame
                # input_embed = torch.nn.utils.rnn.pad_sequence(input_embeds,batch_first=True,padding_value=pad_token_id)
                # cls_label = torch.nn.utils.rnn.pad_sequence(cls_labels,batch_first=True,padding_value=IGNORE_INDEX)

                #mask
                # tf 5.x masking_utils expects 2-D attention_mask [B, T]; the
                # original code used element-wise .ne(pad) on the 3-D embed which
                # produced [B, T, D] and tf 4.x silently coerced. Collapse the
                # hidden dim with .any() to recover [B, T].
                cls_attention_mask = input_embed.ne(pad_token_id).any(dim=-1)

                if cls_training:
                    cls_output= self.cls_net(input_embed,cls_labels=cls_label,cls_attention_mask=cls_attention_mask)
                # cls_loss = self.cls_net(x,None)
                    return cls_output
                else:
                    cls_output= self.cls_net(input_embed,cls_labels=cls_label,cls_attention_mask=cls_attention_mask)

                    return cls_output,cls_label

        if cls_demo:
            pad_token_id = 0
            input_embeds = []
            # start_feature_idx = [0] + frames_features_shape[:-1]
            input_embeds.append(x[0][-1].unsqueeze(0))
            input_embed = torch.nn.utils.rnn.pad_sequence(input_embeds,batch_first=True,padding_value=pad_token_id)
            cls_attention_mask = input_embed.ne(pad_token_id).any(dim=-1)

            # start = time.time()
            cls_output = self.cls_net(input_embed, cls_labels = None, cls_attention_mask = cls_attention_mask)

            return x , cls_output.logits[0][-1]

        return x
