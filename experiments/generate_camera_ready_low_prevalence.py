#!/usr/bin/env python3
"""Reproduce the camera-ready BERT low-prevalence workload table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = q * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    fraction = index - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def percentile(values: list[float], q: float) -> float:
    return quantile(values, q)


def metrics(predictions: list[int], labels: list[int]) -> dict[str, float | int]:
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(predictions, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": tp / (tp + fn),
        "fpr": fp / (fp + tn),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=root / "results/bert_scores_v2.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results/table2_low_prevalence_camera_ready.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "results/low_prevalence_camera_ready_summary.json",
    )
    args = parser.parse_args()

    rows = json.loads(args.scores.read_text())
    val = [r for r in rows if r["split"] == "val"]
    cal = [r for r in rows if r["split"] == "cal"]
    test = [r for r in rows if r["split"] == "test"]
    if (len(val), len(cal), len(test)) != (3742, 2000, 7991):
        raise SystemExit("Unexpected val/cal/test score counts")

    val_legit = [float(r["score_bert"]) for r in val if int(r["label"]) == 0]
    cal_legit = [float(r["score_bert"]) for r in cal if int(r["label"]) == 0]
    test_scores = [float(r["score_bert"]) for r in test]
    labels = [int(r["label"]) for r in test]
    if (len(val_legit), len(cal_legit), labels.count(0), labels.count(1)) != (
        1705,
        2000,
        3911,
        4080,
    ):
        raise SystemExit("Unexpected class counts")

    threshold = quantile(val_legit, 0.99)
    raw = [int(score >= threshold) for score in test_scores]
    conformal = [
        int((1 + sum(cal_score >= score for cal_score in cal_legit)) / 2001 <= 0.01)
        for score in test_scores
    ]
    methods = [
        ("Raw threshold", raw),
        ("Temp. scaling", raw),
        ("Conformal", conformal),
    ]

    legit_indices = [i for i, label in enumerate(labels) if label == 0]
    scale = 9800 / len(legit_indices)
    rng = random.Random(20260527)
    bootstrap = {name: [] for name, _ in methods}
    differences: list[float] = []
    for _ in range(5000):
        sample = [rng.choice(legit_indices) for _ in legit_indices]
        raw_fp = sum(raw[i] for i in sample) * scale
        conformal_fp = sum(conformal[i] for i in sample) * scale
        bootstrap["Raw threshold"].append(raw_fp)
        bootstrap["Temp. scaling"].append(raw_fp)
        bootstrap["Conformal"].append(conformal_fp)
        differences.append(raw_fp - conformal_fp)

    output_rows = []
    for name, predictions in methods:
        result = metrics(predictions, labels)
        fp_per_10k = float(result["fpr"]) * 9800
        tp_per_10k = float(result["recall"]) * 200
        precision = tp_per_10k / (tp_per_10k + fp_per_10k)
        output_rows.append(
            {
                "method": name,
                "recall": f"{float(result['recall']):.10f}",
                "fpr": f"{float(result['fpr']):.10f}",
                "precision_at_2pct": f"{precision:.10f}",
                "fp_per_10k": f"{fp_per_10k:.6f}",
                "fp_ci_low": f"{percentile(bootstrap[name], 0.025):.6f}",
                "fp_ci_high": f"{percentile(bootstrap[name], 0.975):.6f}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    raw_result = metrics(raw, labels)
    conformal_result = metrics(conformal, labels)
    summary = {
        "alpha": 0.01,
        "target_prevalence": 0.02,
        "validation_legitimate": len(val_legit),
        "calibration_legitimate": len(cal_legit),
        "test_legitimate": labels.count(0),
        "test_positive": labels.count(1),
        "raw_threshold": threshold,
        "bootstrap_resamples": 5000,
        "bootstrap_seed": 20260527,
        "raw_false_positives_test": raw_result["fp"],
        "conformal_false_positives_test": conformal_result["fp"],
        "raw_minus_conformal_fp_per_10k": (
            float(raw_result["fpr"]) - float(conformal_result["fpr"])
        )
        * 9800,
        "raw_minus_conformal_ci": [
            percentile(differences, 0.025),
            percentile(differences, 0.975),
        ],
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
