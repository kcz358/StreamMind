import os

import torch
import torch.distributed as dist

from ov_train.evaluate import build, parse_args, relaxed_correct
from streammind.constants import IGNORE_INDEX


def main():
    args = parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    _, model, _, loader = build(args, rank)

    trigger_sum = 0.0
    timeval_sum = 0.0
    count = 0
    for index, inputs in enumerate(loader):
        if index % dist.get_world_size() != dist.get_rank():
            continue
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(rank)
        with torch.no_grad():
            cls_output, cls_label = model(**inputs)
        logits = cls_output.logits[..., :-1, :].reshape(-1, 2)
        labels = cls_label[..., 1:].to(rank).reshape(-1)
        mask = labels != IGNORE_INDEX
        labels = labels[mask]
        predictions = logits[mask].argmax(dim=-1)
        correct = relaxed_correct(labels, predictions, args.tolerance_frames)
        trigger_sum += correct.float().mean().item()
        false_positives = (((labels == 0) & (predictions == 1)) & ~correct).sum()
        false_negatives = (((labels == 1) & (predictions == 0)) & ~correct).sum()
        silence_rate = 1 - false_positives / (labels == 0).sum().clamp_min(1)
        speak_rate = 1 - false_negatives / (labels == 1).sum().clamp_min(1)
        timeval_sum += (silence_rate * speak_rate).item()
        count += 1

    totals = torch.tensor([trigger_sum, timeval_sum, count], dtype=torch.float64, device=rank)
    dist.all_reduce(totals)
    if dist.get_rank() == 0:
        print(f"TriggerAcc: {(totals[0] / totals[2]).item():.6f}")
        print(f"TimVal:     {(totals[1] / totals[2]).item():.6f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
