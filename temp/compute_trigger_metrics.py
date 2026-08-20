import argparse
import glob
import json
from pathlib import Path

import numpy as np


def auc(x, y):
    order = np.argsort(x)
    return float(np.trapz(np.asarray(y)[order], np.asarray(x)[order]))


def relaxed_correct(labels, predictions, tolerance):
    matches = np.zeros(len(labels), dtype=bool)
    for index, label in enumerate(labels):
        start = max(0, index - tolerance)
        end = min(len(labels), index + tolerance + 1)
        matches[index] = label in predictions[start:end]
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--pattern", required=True)
    args = parser.parse_args()

    records = []
    for path in sorted(glob.glob(str(Path(args.directory) / args.pattern))):
        with open(path) as file:
            records.extend(json.loads(line) for line in file if line.strip())

    records.sort(key=lambda record: record["half"])
    labels = np.concatenate([np.asarray(record["labels"], dtype=np.int8) for record in records])
    predictions = np.concatenate([np.asarray(record["predictions"], dtype=np.int8) for record in records])
    probabilities = np.concatenate(
        [np.asarray(record["response_probabilities"], dtype=np.float64) for record in records]
    )

    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    positives = int(labels.sum())
    negatives = len(labels) - positives
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    distinct = np.r_[np.where(np.diff(probabilities[order]))[0], len(labels) - 1]
    tpr = np.r_[0.0, tps[distinct] / max(positives, 1)]
    fpr = np.r_[0.0, fps[distinct] / max(negatives, 1)]
    roc_auc = auc(fpr, tpr)

    pr_precision = tps[distinct] / np.maximum(tps[distinct] + fps[distinct], 1)
    pr_recall = tps[distinct] / max(positives, 1)
    pr_auc = auc(np.r_[0.0, pr_recall], np.r_[1.0, pr_precision])

    trigger_scores = []
    timeval_scores = []
    for record in records:
        half_labels = np.asarray(record["labels"], dtype=np.int8)
        half_predictions = np.asarray(record["predictions"], dtype=np.int8)
        correct = relaxed_correct(half_labels, half_predictions, int(record["tolerance"]))
        trigger_scores.append(float(correct.mean()))
        relaxed_fp = (((half_labels == 0) & (half_predictions == 1)) & ~correct).sum()
        relaxed_fn = (((half_labels == 1) & (half_predictions == 0)) & ~correct).sum()
        silence_rate = 1 - relaxed_fp / max((half_labels == 0).sum(), 1)
        speak_rate = 1 - relaxed_fn / max((half_labels == 1).sum(), 1)
        timeval_scores.append(float(silence_rate * speak_rate))

    metrics = {
        "halves": len(records),
        "positions": int(len(labels)),
        "positives": positives,
        "negatives": negatives,
        "tolerance": int(records[0]["tolerance"]),
        "trigger_acc_macro": float(np.mean(trigger_scores)),
        "timval_macro": float(np.mean(timeval_scores)),
        "accuracy_micro": float((tp + tn) / len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "specificity": tn / max(tn + fp, 1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    output = Path(args.directory) / "metrics.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
