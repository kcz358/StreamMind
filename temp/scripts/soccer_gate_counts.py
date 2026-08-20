import argparse

import torch

from ov_train.evaluate import build, relaxed_correct
from streammind.constants import IGNORE_INDEX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--resume_from", required=True)
    parser.add_argument("--data_type", default="valid")
    parser.add_argument("--tolerance_frames", type=int, default=2)
    parser.add_argument("--max_halves", type=int, default=-1)
    parser.add_argument("--eval_type", default="cls")
    parser.add_argument("--caption_csv", default="")
    args = parser.parse_args()

    _, model, _, loader = build(args, "cuda")
    totals = {
        "label_silence": 0,
        "label_speak": 0,
        "pred_silence": 0,
        "pred_speak": 0,
        "raw_tn": 0,
        "raw_fp": 0,
        "raw_fn": 0,
        "raw_tp": 0,
        "relaxed_correct": 0,
        "relaxed_fp": 0,
        "relaxed_fn": 0,
    }
    macro_trigger = []
    macro_timval = []

    for half, inputs in enumerate(loader, 1):
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(model.model.ov_model.device)
        with torch.no_grad():
            cls_output, cls_label = model(**inputs)

        logits = cls_output.logits[..., :-1, :].reshape(-1, 2)
        labels = cls_label[..., 1:].reshape(-1).to(logits.device)
        mask = labels != IGNORE_INDEX
        labels = labels[mask]
        preds = logits[mask].float().softmax(dim=-1).argmax(dim=-1)
        relaxed = relaxed_correct(labels, preds, args.tolerance_frames)

        label_silence = (labels == 0).sum().item()
        label_speak = (labels == 1).sum().item()
        raw_tn = ((labels == 0) & (preds == 0)).sum().item()
        raw_fp = ((labels == 0) & (preds == 1)).sum().item()
        raw_fn = ((labels == 1) & (preds == 0)).sum().item()
        raw_tp = ((labels == 1) & (preds == 1)).sum().item()
        relaxed_fp = (((labels == 0) & (preds == 1)) & ~relaxed).sum().item()
        relaxed_fn = (((labels == 1) & (preds == 0)) & ~relaxed).sum().item()

        totals["label_silence"] += label_silence
        totals["label_speak"] += label_speak
        totals["pred_silence"] += (preds == 0).sum().item()
        totals["pred_speak"] += (preds == 1).sum().item()
        totals["raw_tn"] += raw_tn
        totals["raw_fp"] += raw_fp
        totals["raw_fn"] += raw_fn
        totals["raw_tp"] += raw_tp
        totals["relaxed_correct"] += relaxed.sum().item()
        totals["relaxed_fp"] += relaxed_fp
        totals["relaxed_fn"] += relaxed_fn

        trigger = relaxed.float().mean().item()
        silence_rate = 1 - relaxed_fp / max(label_silence, 1)
        speak_rate = 1 - relaxed_fn / max(label_speak, 1)
        macro_trigger.append(trigger)
        macro_timval.append(silence_rate * speak_rate)

        if args.max_halves >= 0 and half >= args.max_halves:
            break

    valid = totals["label_silence"] + totals["label_speak"]
    silence_rate = 1 - totals["relaxed_fp"] / totals["label_silence"]
    speak_rate = 1 - totals["relaxed_fn"] / totals["label_speak"]
    print("COUNTS", totals)
    print("VALID_POSITIONS", valid)
    print("LABEL_SPEAK_RATIO", totals["label_speak"] / valid)
    print("PRED_SPEAK_RATIO", totals["pred_speak"] / valid)
    print("RAW_ACCURACY", (totals["raw_tn"] + totals["raw_tp"]) / valid)
    print("MICRO_RELAXED_TRIGGER_ACC", totals["relaxed_correct"] / valid)
    print("MICRO_RELAXED_SILENCE_RATE", silence_rate)
    print("MICRO_RELAXED_SPEAK_RATE", speak_rate)
    print("MICRO_TIMVAL_PRODUCT", silence_rate * speak_rate)
    print("MACRO_TRIGGER_ACC", sum(macro_trigger) / len(macro_trigger))
    print("MACRO_TIMVAL", sum(macro_timval) / len(macro_timval))


if __name__ == "__main__":
    main()
