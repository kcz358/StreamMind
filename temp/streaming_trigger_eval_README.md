# Streaming Trigger Evaluation Logs

Evaluation date: 2026-07-23

## Directories

- `mage_strict/`: Mage-VL-Base optional gate, exact position matching
  (`tolerance=0`). Per-rank records are in `rank_*.jsonl`.
- `joyai_pm1/`: JoyAI-VL-Interaction, one-second decision positions with
  `tolerance=1`. Per-rank records are in `records_*.jsonl`.

Each half record contains `labels`, argmax `predictions`, and
`response_probabilities`. The arrays have identical lengths.

## Metrics

- `trigger_acc_macro` and `timval_macro` use the model-specific tolerance above
  and are macro-averaged over 98 match halves.
- Precision, recall, F1, specificity, accuracy, and confusion matrix use raw
  per-position argmax predictions without tolerance relaxation.
- ROC-AUC and PR-AUC use the saved response probabilities.

Each model directory contains its aggregate `metrics.json` and full run log.
