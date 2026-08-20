import gc
import os
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

sys.path.insert(0, "/root/StreamMind")
from ov_train.onevision_stream import OneVisionStreamForCausalLM
from ov_train.stream_dataset import DataArguments, LazySupervisedDataset
from ov_train.train import DataCollatorForstreamDataset, _resolve_image_processor


base = "/data/kaichen/checkpoints/OV2/Stage5_SFT_4B_mimo_v15_lr2e-6_HF_ported"
checkpoint = (
    "/data/kaichen/StreamMind/p16_4b_stage2_train_all_20260715-121029/"
    "checkpoint-417994/model.safetensors"
)
release = "/tmp/mage_vl_streammind_release"

tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
args = DataArguments()
args.soccer_dataset = True
args.soccer_dataset_train_cls = True
args.video_backend = "codec"
args.data_type = "valid"
args.video_processor = _resolve_image_processor(processor)
args.image_processor = args.video_processor
args.processor = processor
args.is_multimodal = True
args.cur_fps = 2
dataset = LazySupervisedDataset(data_path="", tokenizer=tokenizer, data_args=args)
batch = DataCollatorForstreamDataset(tokenizer=tokenizer)([dataset[0]])
segments = batch["images"][0]
for segment in segments:
    for key in ("pixel_values", "image_grid_thw", "patch_positions"):
        segment[key] = segment[key].cuda()

old = OneVisionStreamForCausalLM(base).cuda().eval()
old.add_streammind_special_tokens(tokenizer)
old.load_state_dict(load_file(checkpoint), strict=False)
old_attn = old.model.ov_model.model.visual.config._attn_implementation
old_dtype = next(old.model.ov_model.model.visual.parameters()).dtype
old_rope = old.model.ov_model.model.visual.video_rope.inv_freq_t.detach().cpu()

with torch.no_grad():
    old_vision_parts = [old._encode_frames_with_onevision(segment) for segment in segments]
    old_vision = torch.cat(old_vision_parts, dim=1)
    old_perception = old.model.mm_projector(old_vision)
    boundaries = torch.tensor([part.shape[1] for part in old_vision_parts]).cumsum(0).tolist()
    old_gate, _ = old.model.mm_projector(
        old_vision,
        cls_inference=True,
        frames_features_shape=boundaries,
        prompt_time_input_ids=batch["input_ids"].cuda(),
        prompt_time_lable=batch["labels"],
    )
    old_logits = old_gate.logits[:, 0].reshape(1, -1, 2)
    old_gate_model = old.model.mm_projector.cls_net.cls_model
    old_target_embed = old_gate_model.model.embed_tokens(
        torch.tensor([0, 1], device="cuda")
    ).detach().cpu()
    old_gate_config = old_gate_model.config.to_dict()
    old_gate_rope = old_gate_model.model.rotary_emb.inv_freq.detach().cpu()
    old_gate_state = {
        key: value.detach().cpu()
        for key, value in old_gate_model.state_dict().items()
    }

old_vision = old_vision.cpu()
old_perception = old_perception.cpu()
old_logits = old_logits.cpu()
del old
gc.collect()
torch.cuda.empty_cache()

new = AutoModelForImageTextToText.from_pretrained(
    release, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
).cuda().eval()
print("old_attn", old_attn)
print("new_attn", new.model.visual.config._attn_implementation)
print("old_dtype", old_dtype)
print("new_dtype", next(new.model.visual.parameters()).dtype)
print("rope_max_abs", (old_rope - new.model.visual.video_rope.inv_freq_t.detach().cpu()).abs().max().item())
print("old_rope", old_rope.tolist())
print("new_rope", new.model.visual.video_rope.inv_freq_t.detach().cpu().tolist())
print("old_gate_rope_dtype", old_gate_rope.dtype)
with torch.no_grad():
    new_vision_parts = [
        new.model._streammind_vision_tokens(
            segment["pixel_values"], segment["image_grid_thw"], segment["patch_positions"]
        )
        for segment in segments
    ]
    new_vision = torch.cat(new_vision_parts, dim=1)
    new_perception = new.model.streammind_gate.perception_tokens(new_vision)
    new_logits = new.model.streammind_gate(new_vision, response_positions=boundaries)
    new_gate_model = new.model.streammind_gate.cls_net.cls_model
    new_target_embed = new_gate_model.model.embed_tokens(
        torch.tensor([0, 1], device="cuda")
    ).detach().cpu()
    new_gate_config = new_gate_model.config.to_dict()
    new_gate_rope = new_gate_model.model.rotary_emb.inv_freq.detach().cpu()
print("new_gate_rope_dtype", new_gate_rope.dtype)
print("old_gate_rope", old_gate_rope.tolist())
print("new_gate_rope", new_gate_rope.tolist())

for key in sorted(set(old_gate_config) | set(new_gate_config)):
    if old_gate_config.get(key) != new_gate_config.get(key):
        print("CONFIG_DIFF", key, old_gate_config.get(key), new_gate_config.get(key))

for key, old_value in old_gate_state.items():
    new_value = new_gate_model.state_dict()[key].detach().cpu()
    if not torch.equal(old_value, new_value):
        print("PARAM_DIFF", key, (old_value.float() - new_value.float()).abs().max().item())
        break
else:
    print("ALL_GATE_PARAMS_EQUAL", len(old_gate_state))


def report(name, left, right):
    left = left.float()
    right = right.detach().cpu().float()
    print(name, "shape", tuple(left.shape), tuple(right.shape))
    print(name, "max_abs", (left - right).abs().max().item())
    print(name, "mean_abs", (left - right).abs().mean().item())


report("vision", old_vision, new_vision)
report("perception", old_perception, new_perception)
report("target_embed", old_target_embed, new_target_embed)
report("gate_rope", old_gate_rope, new_gate_rope)
report("logits", old_logits, new_logits)
