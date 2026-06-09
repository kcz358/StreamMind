"""Compute BLEU-1/BLEU-4/METEOR/ROUGE-L from a (pred, target) caption CSV.

Mirrors paper's `streammind/eval/score_single.py` / `get_csv_caption_evaluation_metric.py`
but standalone: no streammind imports, uses the same `pycocoevalcap` scorers
SoccerNet uses.

Install on the pod:
  pip install pycocoevalcap

Usage:
  python -m ov_train.nlg_eval --csv /tmp/eval_captions_stage1.csv
"""
from __future__ import annotations

import argparse
import csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with header [pred,target]")
    p.add_argument("--max_rows", type=int, default=-1)
    return p.parse_args()


def load_csv(path: str, max_rows: int):
    refs = {}
    hyps = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for i, row in enumerate(reader):
            if max_rows >= 0 and i >= max_rows:
                break
            if len(row) < 2:
                continue
            pred, target = row[0].strip(), row[1].strip()
            if not pred and not target:
                continue
            hyps[i] = [pred]
            refs[i] = [target]
    return refs, hyps


def compute_metrics(refs: dict, hyps: dict) -> dict:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    bleu4_scorer = Bleu(4)
    meteor_scorer = Meteor()
    rouge_scorer = Rouge()

    bleu4_score, _ = bleu4_scorer.compute_score(refs, hyps)
    meteor_score, _ = meteor_scorer.compute_score(refs, hyps)
    _, rouge_scores = rouge_scorer.compute_score(refs, hyps)
    rouge_l_score = sum(rouge_scores) / len(rouge_scores)

    return {
        "BLEU-1": bleu4_score[0] * 100,
        "BLEU-2": bleu4_score[1] * 100,
        "BLEU-3": bleu4_score[2] * 100,
        "BLEU-4": bleu4_score[3] * 100,
        "METEOR": meteor_score * 100,
        "ROUGE-L": rouge_l_score * 100,
    }


def main():
    args = parse_args()
    refs, hyps = load_csv(args.csv, args.max_rows)
    print(f"[nlg_eval] {len(refs)} caption pairs from {args.csv}")
    metrics = compute_metrics(refs, hyps)
    print(f"\n[nlg_eval] SoccerNet-style metrics:")
    for k, v in metrics.items():
        print(f"  {k:8s} {v:.2f}")


if __name__ == "__main__":
    main()
