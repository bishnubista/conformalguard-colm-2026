#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "numpy>=1.26",
#   "scipy>=1.12",
#   "scikit-learn>=1.3",
# ]
# ///
"""
Additional experiments for the ConformalGuard paper:
1. Calibration set sensitivity (BERT + conformal, n=100,500,1000,5000,full)
2. Metric verification for all results
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def calibrate_conformal(cal_scores_legit, test_scores, alpha):
    """Conformal prediction using legitimate-only calibration set."""
    n = len(cal_scores_legit)
    pvalues = np.array([
        (1 + np.sum(cal_scores_legit >= s)) / (n + 1)
        for s in test_scores
    ])
    return (pvalues <= alpha).astype(int)


def compute_metrics(predictions, labels, alpha):
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "fpr": round(fpr, 4),
        "f1": round(f1, 4),
        "fpr_violation": round(fpr - alpha, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def run_cal_sensitivity(df, alpha=0.01, n_repeats=20, seed=42):
    """Subsample calibration set at various sizes, report FPR and recall."""
    rng = np.random.RandomState(seed)

    cal_df = df[(df["split"] == "cal") & (df["label"] == 0)]
    test_df = df[df["split"] == "test"]
    cal_scores_full = cal_df["score_bert"].values.astype(float)
    test_scores = test_df["score_bert"].values.astype(float)
    test_labels = test_df["label"].values.astype(int)

    sizes = [100, 500, 1000, 5000, len(cal_scores_full)]
    results = []

    for n_cal in sizes:
        if n_cal > len(cal_scores_full):
            n_cal = len(cal_scores_full)

        fpr_list, recall_list = [], []
        for _ in range(n_repeats if n_cal < len(cal_scores_full) else 1):
            if n_cal < len(cal_scores_full):
                idx = rng.choice(len(cal_scores_full), size=n_cal, replace=False)
                cal_sub = cal_scores_full[idx]
            else:
                cal_sub = cal_scores_full

            preds = calibrate_conformal(cal_sub, test_scores, alpha)
            m = compute_metrics(preds, test_labels, alpha)
            fpr_list.append(m["fpr"])
            recall_list.append(m["recall"])

        results.append({
            "n_cal": n_cal,
            "fpr_mean": round(np.mean(fpr_list), 4),
            "fpr_std": round(np.std(fpr_list), 4),
            "recall_mean": round(np.mean(recall_list), 4),
            "recall_std": round(np.std(recall_list), 4),
        })
        print(f"  n_cal={n_cal:5d}: FPR={np.mean(fpr_list):.4f}±{np.std(fpr_list):.4f}, "
              f"Recall={np.mean(recall_list):.4f}±{np.std(recall_list):.4f}")

    return pd.DataFrame(results)


def verify_metrics(results_path):
    """Verify arithmetic consistency of all results."""
    with open(results_path) as f:
        results = json.load(f)

    issues = []
    for r in results:
        tp, fp, tn, fn = r["tp"], r["fp"], r["tn"], r["fn"]

        # Verify precision
        expected_prec = tp / max(tp + fp, 1)
        if abs(r["precision"] - round(expected_prec, 4)) > 0.0001:
            issues.append(f"{r['lm']}|{r['calibration']}: precision mismatch {r['precision']} vs {expected_prec:.4f}")

        # Verify recall
        expected_rec = tp / max(tp + fn, 1)
        if abs(r["recall"] - round(expected_rec, 4)) > 0.0001:
            issues.append(f"{r['lm']}|{r['calibration']}: recall mismatch {r['recall']} vs {expected_rec:.4f}")

        # Verify FPR
        expected_fpr = fp / max(fp + tn, 1)
        if abs(r["fpr"] - round(expected_fpr, 4)) > 0.0001:
            issues.append(f"{r['lm']}|{r['calibration']}: FPR mismatch {r['fpr']} vs {expected_fpr:.4f}")

        # Verify F1
        if r["precision"] + r["recall"] > 0:
            expected_f1 = 2 * r["precision"] * r["recall"] / (r["precision"] + r["recall"])
            if abs(r["f1"] - round(expected_f1, 4)) > 0.0001:
                issues.append(f"{r['lm']}|{r['calibration']}: F1 mismatch {r['f1']} vs {expected_f1:.4f}")

        # Verify violation
        expected_viol = r["fpr"] - r["alpha"]
        if abs(r["fpr_violation"] - round(expected_viol, 4)) > 0.0001:
            issues.append(f"{r['lm']}|{r['calibration']}: violation mismatch")

    return issues


def main():
    print("=" * 60)
    print("ADDITIONAL EXPERIMENTS")
    print("=" * 60)

    df = pd.read_csv("scores.csv")

    # 1. Calibration sensitivity
    print("\n## Calibration Set Sensitivity (BERT + Conformal)")
    cal_sens = run_cal_sensitivity(df, alpha=0.01)
    cal_sens.to_csv("results/calibration_sensitivity.csv", index=False)
    print(f"\nSaved to results/calibration_sensitivity.csv")

    # 2. Metric verification
    print("\n## Metric Verification")
    issues = verify_metrics("results/metric_verification.json")
    if issues:
        print(f"  ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ALL METRICS VERIFIED CORRECT")

    # Save verification result
    verification = {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "n_rows_checked": len(json.load(open("results/metric_verification.json"))),
    }
    with open("results/metric_verification_result.json", "w") as f:
        json.dump(verification, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
