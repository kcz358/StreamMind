import os
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

sys.path.insert(0, "/root/StreamMind")
from ov_train.evaluate import relaxed_correct
from ov_train.stream_dataset import DataArguments, LazySupervisedDataset
from ov_train.train import DataCollatorForstreamDataset, _resolve_image_processor
from streammind.constants import IGNORE_INDEX


root = "/tmp/mage_vl_streammind_release"
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)
tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(root, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    root,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(local_rank).eval()

data_args = DataArguments()
data_args.soccer_dataset = True
data_args.soccer_dataset_train_cls = True
data_args.soccer_dataset_train_llm = False
data_args.video_backend = "codec"
data_args.data_type = "valid"
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

    boundaries = torch.tensor([segment["image_grid_thw"][:, 0].sum() for segment in segments]).cumsum(0)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    labels[boundaries.to(logits.device) - 1] = 1
    preds = logits.argmax(dim=-1)
    relaxed = relaxed_correct(labels, preds, 2)
    trigger = relaxed.float().mean().item()
    fp = (((labels == 0) & (preds == 1)) & ~relaxed).sum().item()
    fn = (((labels == 1) & (preds == 0)) & ~relaxed).sum().item()
    silence_rate = 1 - fp / max((labels == 0).sum().item(), 1)
    speak_rate = 1 - fn / max((labels == 1).sum().item(), 1)
    trigger_scores.append(trigger)
    timval_scores.append(silence_rate * speak_rate)
    print(f"[rank {dist.get_rank()} half {half}] TriggerAcc={trigger:.4f} TimVal={timval_scores[-1]:.4f}")

totals = torch.tensor(
    [sum(trigger_scores), sum(timval_scores), len(trigger_scores)],
    dtype=torch.float64,
    device=local_rank,
)
dist.all_reduce(totals)
if dist.get_rank() == 0:
    print("TriggerAcc", (totals[0] / totals[2]).item())
    print("TimVal", (totals[1] / totals[2]).item())
dist.destroy_process_group()
