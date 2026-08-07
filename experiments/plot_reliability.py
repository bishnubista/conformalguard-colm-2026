#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "numpy>=1.26",
#   "scipy>=1.12",
#   "scikit-learn>=1.3",
#   "matplotlib>=3.8",
# ]
# ///
"""
Generate a clean calibration reliability plot with one panel per LM.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression


def calibrate_raw_threshold(val_scores, val_labels, test_scores, alpha):
    val_neg = val_scores[val_labels == 0]
    threshold = np.quantile(val_neg, 1 - alpha)
    return (test_scores >= threshold).astype(int)


def calibrate_temperature(val_scores, val_labels, test_scores, alpha):
    eps = 1e-7
    val_s = np.clip(val_scores, eps, 1 - eps)

    def nll(T):
        logits = np.log(val_s / (1 - val_s)) / T
        probs = 1 / (1 + np.exp(-logits))
        return -np.mean(val_labels * np.log(probs + eps) + (1 - val_labels) * np.log(1 - probs + eps))

    T = minimize_scalar(nll, bounds=(0.01, 20.0), method="bounded").x
    test_s = np.clip(test_scores, eps, 1 - eps)
    test_cal = 1 / (1 + np.exp(-np.log(test_s / (1 - test_s)) / T))
    val_cal = 1 / (1 + np.exp(-np.log(val_s / (1 - val_s)) / T))
    threshold = np.quantile(val_cal[val_labels == 0], 1 - alpha)
    return (test_cal >= threshold).astype(int)


def calibrate_isotonic(val_scores, val_labels, test_scores, alpha):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(val_scores, val_labels)
    test_cal = iso.predict(test_scores)
    val_cal = iso.predict(val_scores)
    threshold = np.quantile(val_cal[val_labels == 0], 1 - alpha)
    return (test_cal >= threshold).astype(int)


def calibrate_conformal(cal_legit, test_scores, alpha):
    n = len(cal_legit)
    pvalues = np.array([(1 + np.sum(cal_legit >= s)) / (n + 1) for s in test_scores])
    return (pvalues <= alpha).astype(int)


def compute_fpr(preds, labels):
    fp = np.sum((preds == 1) & (labels == 0))
    tn = np.sum((preds == 0) & (labels == 0))
    return fp / max(fp + tn, 1)


def main():
    df = pd.read_csv("scores.csv")
    val_df = df[df["split"] == "val"]
    cal_df = df[(df["split"] == "cal") & (df["label"] == 0)]
    test_df = df[df["split"] == "test"]

    alphas = [0.005, 0.01, 0.02, 0.05, 0.10]
    lms = {"BERT": "score_bert", "Claude Sonnet 4": "score_claude", "GPT-4o": "score_gpt"}

    methods = {
        "Raw threshold": {"marker": "o", "color": "#E24B4A", "ls": "-"},
        "Temp. scaling": {"marker": "s", "color": "#BA7517", "ls": "--"},
        "Isotonic reg.": {"marker": "^", "color": "#378ADD", "ls": "-."},
        "Conformal": {"marker": "D", "color": "#1D9E75", "ls": "-"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax_idx, (lm_name, score_col) in enumerate(lms.items()):
        ax = axes[ax_idx]

        val_scores = val_df[score_col].values.astype(float)
        val_labels = val_df["label"].values.astype(int)
        cal_scores = cal_df[score_col].values.astype(float)
        test_scores = test_df[score_col].values.astype(float)
        test_labels = test_df["label"].values.astype(int)

        val_scores = np.nan_to_num(val_scores, nan=0.5)
        cal_scores = np.nan_to_num(cal_scores, nan=0.5)
        test_scores = np.nan_to_num(test_scores, nan=0.5)

        for method_name, style in methods.items():
            fprs = []
            for alpha in alphas:
                if method_name == "Raw threshold":
                    preds = calibrate_raw_threshold(val_scores, val_labels, test_scores, alpha)
                elif method_name == "Temp. scaling":
                    preds = calibrate_temperature(val_scores, val_labels, test_scores, alpha)
                elif method_name == "Isotonic reg.":
                    preds = calibrate_isotonic(val_scores, val_labels, test_scores, alpha)
                elif method_name == "Conformal":
                    preds = calibrate_conformal(cal_scores, test_scores, alpha)
                fprs.append(compute_fpr(preds, test_labels))

            ax.plot(alphas, fprs, marker=style["marker"], color=style["color"],
                    linestyle=style["ls"], label=method_name, markersize=7, linewidth=1.8)

        # Diagonal
        ax.plot([0, 0.12], [0, 0.12], "k--", alpha=0.3, linewidth=1, label="Perfect")
        ax.set_xlabel("Target FPR (α)", fontsize=12)
        ax.set_title(lm_name, fontsize=13, fontweight="bold")
        ax.set_xlim(0, 0.11)
        ax.set_ylim(0, 0.16)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel("Observed FPR", fontsize=12)
    fig.suptitle("Calibration Reliability: Observed FPR vs Target α", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("results/calibration_reliability_plot.png", dpi=300, bbox_inches="tight")
    print("Saved results/calibration_reliability_plot.png")


if __name__ == "__main__":
    main()
