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
ConformalGuard: Calibration Method Comparison Across LM Classifiers
====================================================================

Produces Table 1 of the paper: 3 LMs × 4 calibration methods.

This is the central experiment. It answers:
  "Does conformal calibration provide better deployment-time FPR control
   for LM classifiers than standard calibration methods?"

Usage:
    python run_calibration_comparison.py \
        --scores-csv scores.csv \
        --output-dir results/ \
        --alpha 0.01

Input: scores.csv with columns:
    - split: train/val/cal/test
    - label: 0=legitimate, 1=phishing
    - score_bert: BERT P(phishing) ∈ [0,1]
    - score_claude: Claude phishing probability ∈ [0,1]
    - score_gpt: GPT-4o-mini phishing probability ∈ [0,1]
    - source: (optional) corpus origin for diagnostics

Output:
    - results/table1_calibration_comparison.csv
    - results/table1_calibration_comparison.tex  (LaTeX-ready)
    - results/table2_low_prevalence.csv
    - results/calibration_reliability_plot.png
    - results/metric_verification.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# Optional imports
try:
    from sklearn.isotonic import IsotonicRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def log(msg):
    print(f"[experiment] {msg}", flush=True)


# ================================================================
# CALIBRATION METHODS
# ================================================================

def calibrate_raw_threshold(val_scores, val_labels, test_scores, alpha):
    """
    Raw threshold: pick the threshold on val set that achieves FPR ≤ alpha.
    Apply that fixed threshold to test set.
    """
    val_neg_scores = val_scores[val_labels == 0]
    # Threshold = quantile of legitimate scores such that FPR ≤ alpha
    threshold = np.quantile(val_neg_scores, 1 - alpha)
    test_predictions = (test_scores >= threshold).astype(int)
    return test_predictions, {"method": "raw_threshold", "threshold": float(threshold)}


def calibrate_temperature_scaling(val_scores, val_labels, test_scores, alpha):
    """
    Temperature scaling: learn T on validation set, apply to test scores,
    then threshold at the validation quantile matching alpha.
    """
    # Clip scores to avoid log(0)
    eps = 1e-7
    val_s = np.clip(val_scores, eps, 1 - eps)

    def nll(T):
        logits = np.log(val_s / (1 - val_s)) / T
        probs = 1 / (1 + np.exp(-logits))
        loss = -np.mean(
            val_labels * np.log(probs + eps) +
            (1 - val_labels) * np.log(1 - probs + eps)
        )
        return loss

    result = minimize_scalar(nll, bounds=(0.01, 20.0), method="bounded")
    T = result.x

    # Apply temperature to test scores
    test_s = np.clip(test_scores, eps, 1 - eps)
    test_logits = np.log(test_s / (1 - test_s)) / T
    test_calibrated = 1 / (1 + np.exp(-test_logits))

    # Threshold: find the threshold on calibrated val scores that gives FPR ≤ alpha
    val_logits = np.log(val_s / (1 - val_s)) / T
    val_calibrated = 1 / (1 + np.exp(-val_logits))
    val_neg_calibrated = val_calibrated[val_labels == 0]
    threshold = np.quantile(val_neg_calibrated, 1 - alpha)

    test_predictions = (test_calibrated >= threshold).astype(int)
    return test_predictions, {"method": "temp_scaling", "T": float(T), "threshold": float(threshold)}


def calibrate_isotonic(val_scores, val_labels, test_scores, alpha):
    """
    Isotonic regression: fit on validation set, apply to test scores,
    then threshold at the validation quantile matching alpha.
    """
    if not HAS_SKLEARN:
        log("  WARNING: sklearn not available, skipping isotonic")
        return np.zeros(len(test_scores), dtype=int), {"method": "isotonic", "error": "sklearn missing"}

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(val_scores, val_labels)

    test_calibrated = iso.predict(test_scores)
    val_calibrated = iso.predict(val_scores)

    val_neg_calibrated = val_calibrated[val_labels == 0]
    threshold = np.quantile(val_neg_calibrated, 1 - alpha)

    test_predictions = (test_calibrated >= threshold).astype(int)
    return test_predictions, {"method": "isotonic", "threshold": float(threshold)}


def calibrate_conformal(cal_scores_legit, test_scores, alpha):
    """
    Conformal prediction: compute p-values using legitimate-only calibration set.
    Threshold at alpha directly.
    """
    n = len(cal_scores_legit)
    pvalues = np.array([
        (1 + np.sum(cal_scores_legit >= s)) / (n + 1)
        for s in test_scores
    ])
    test_predictions = (pvalues <= alpha).astype(int)
    return test_predictions, {"method": "conformal", "n_cal": n, "alpha": alpha, "pvalues": pvalues}


# ================================================================
# EVALUATION
# ================================================================

def compute_metrics(predictions, labels, alpha):
    """Compute all metrics + verify arithmetic consistency."""
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    fpr_violation = fpr - alpha

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "fpr": round(fpr, 4),
        "f1": round(f1, 4),
        "fpr_violation": round(fpr_violation, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total_pos": tp + fn,
        "total_neg": fp + tn,
    }


def bootstrap_ci(predictions, labels, alpha, n_boot=1000, seed=42):
    """Bootstrap 95% confidence intervals."""
    rng = np.random.RandomState(seed)
    metrics_boot = []

    for _ in range(n_boot):
        idx = rng.choice(len(labels), size=len(labels), replace=True)
        m = compute_metrics(predictions[idx], labels[idx], alpha)
        metrics_boot.append(m)

    ci = {}
    for key in ["recall", "precision", "fpr", "f1", "fpr_violation"]:
        vals = [m[key] for m in metrics_boot]
        ci[key] = {
            "mean": round(np.mean(vals), 4),
            "ci_lo": round(np.percentile(vals, 2.5), 4),
            "ci_hi": round(np.percentile(vals, 97.5), 4),
            "std": round(np.std(vals), 4),
        }
    return ci


# ================================================================
# MAIN EXPERIMENT
# ================================================================

def run_comparison(df, alpha=0.01, n_boot=1000):
    """
    Run the full 3×4 calibration comparison.
    Returns a list of result dicts for Table 1.
    """
    # Identify splits
    val_df = df[df["split"] == "val"].copy()
    cal_df = df[df["split"] == "cal"].copy()
    test_df = df[df["split"] == "test"].copy()

    log(f"Val: {len(val_df)} | Cal: {len(cal_df)} (legit only: {(cal_df['label']==0).sum()}) | Test: {len(test_df)}")

    # Score columns
    score_cols = {
        "BERT (fine-tuned)": "score_bert",
        "Claude Sonnet 4": "score_claude",
        "GPT-4o": "score_gpt",
    }

    # Check which score columns exist
    available = {name: col for name, col in score_cols.items() if col in df.columns}
    if not available:
        log("ERROR: No score columns found. Expected: score_bert, score_claude, score_gpt")
        log(f"Available columns: {list(df.columns)}")
        return []

    log(f"Available scorers: {list(available.keys())}")

    results = []

    for lm_name, score_col in available.items():
        log(f"\n{'='*60}")
        log(f"  LM: {lm_name} (column: {score_col})")
        log(f"{'='*60}")

        val_scores = val_df[score_col].values.astype(float)
        val_labels = val_df["label"].values.astype(int)
        cal_scores = cal_df[cal_df["label"] == 0][score_col].values.astype(float)
        test_scores = test_df[score_col].values.astype(float)
        test_labels = test_df["label"].values.astype(int)

        # Skip if too many NaN
        nan_pct = np.isnan(test_scores).mean()
        if nan_pct > 0.1:
            log(f"  WARNING: {nan_pct:.1%} NaN in test scores — skipping")
            continue

        # Fill NaN with 0.5 (neutral score)
        test_scores = np.nan_to_num(test_scores, nan=0.5)
        val_scores = np.nan_to_num(val_scores, nan=0.5)
        cal_scores = np.nan_to_num(cal_scores, nan=0.5)

        methods = [
            ("Raw threshold", lambda vs=val_scores, vl=val_labels, ts=test_scores, a=alpha: calibrate_raw_threshold(vs, vl, ts, a)),
            ("Temp. scaling", lambda vs=val_scores, vl=val_labels, ts=test_scores, a=alpha: calibrate_temperature_scaling(vs, vl, ts, a)),
            ("Isotonic reg.", lambda vs=val_scores, vl=val_labels, ts=test_scores, a=alpha: calibrate_isotonic(vs, vl, ts, a)),
            ("Conformal", lambda cs=cal_scores, ts=test_scores, a=alpha: calibrate_conformal(cs, ts, a)),
        ]

        for method_name, calibrate_fn in methods:
            log(f"  {method_name}...")
            predictions, info = calibrate_fn()
            metrics = compute_metrics(predictions, test_labels, alpha)
            ci = bootstrap_ci(predictions, test_labels, alpha, n_boot)

            result = {
                "lm": lm_name,
                "calibration": method_name,
                "alpha": alpha,
                **metrics,
                "ci": ci,
                "info": {k: v for k, v in info.items() if k != "pvalues"},
            }

            log(f"    Recall={metrics['recall']:.4f}  FPR={metrics['fpr']:.4f}  "
                f"Violation={metrics['fpr_violation']:+.4f}  "
                f"Precision={metrics['precision']:.4f}  F1={metrics['f1']:.4f}")
            log(f"    TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")

            results.append(result)

    return results


def format_table_latex(results, alpha):
    """Generate LaTeX table for the paper."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{FPR control across three LMs and four calibration methods ($\alpha = " + f"{alpha}" + r"$)}",
        r"\label{tab:calibration_comparison}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"LM & Calibration & Recall & Obs.\ FPR & FPR viol. & Precision & F1 \\",
        r"\midrule",
    ]

    current_lm = None
    for r in results:
        lm_display = r["lm"] if r["lm"] != current_lm else ""
        current_lm = r["lm"]

        ci = r["ci"]
        bold = r"\textbf{" if r["calibration"] == "Conformal" else ""
        unbold = "}" if r["calibration"] == "Conformal" else ""

        viol = r["fpr_violation"]
        viol_str = f"{viol:+.4f}"

        lines.append(
            f"{lm_display} & {bold}{r['calibration']}{unbold} & "
            f"{bold}{r['recall']:.3f} $\\pm$ {ci['recall']['std']:.3f}{unbold} & "
            f"{bold}{r['fpr']:.3f}{unbold} & "
            f"{bold}{viol_str}{unbold} & "
            f"{bold}{r['precision']:.3f}{unbold} & "
            f"{bold}{r['f1']:.3f}{unbold} \\\\"
        )

        # Add midrule between LMs
        if r == results[-1]:
            pass
        elif results[results.index(r) + 1]["lm"] != current_lm:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def plot_calibration_reliability(results, alphas, output_dir):
    """
    Plot observed FPR vs target alpha for each calibration method.
    The paper's most important figure.
    """
    if not HAS_MPL:
        log("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    method_styles = {
        "Raw threshold": ("o", "#E24B4A", "-"),
        "Temp. scaling": ("s", "#BA7517", "--"),
        "Isotonic reg.": ("^", "#378ADD", "-."),
        "Conformal": ("D", "#1D9E75", "-"),
    }

    for method_name, (marker, color, ls) in method_styles.items():
        method_results = [r for r in results if r["calibration"] == method_name]
        if not method_results:
            continue

        # Aggregate observed FPR per alpha across all LMs (avoids zig-zagging curves)
        grouped: dict = {}
        for r in method_results:
            grouped.setdefault(r["alpha"], []).append(r["fpr"])
        xs = sorted(grouped)
        ys = [float(np.mean(grouped[a])) for a in xs]

        ax.plot(xs, ys, marker=marker, color=color, linestyle=ls,
                label=method_name, markersize=6, linewidth=1.5)

    # Diagonal (perfect calibration)
    ax.plot([0, max(alphas) * 1.1], [0, max(alphas) * 1.1],
            "k--", alpha=0.3, linewidth=1, label="Perfect")

    ax.set_xlabel("Target FPR (α)", fontsize=12)
    ax.set_ylabel("Observed FPR", fontsize=12)
    ax.set_title("Calibration reliability across methods", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, max(alphas) * 1.1)
    ax.set_ylim(0, max(alphas) * 1.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "calibration_reliability_plot.png", dpi=300, bbox_inches="tight")
    log(f"Saved calibration reliability plot to {output_dir}/calibration_reliability_plot.png")


def run_low_prevalence(df, best_lm_col, alpha=0.01, target_prev=0.02, n_repeats=50, seed=42):
    """
    Simulate low prevalence by subsampling positives.
    Run all calibration methods on the best LM.
    """
    rng = np.random.RandomState(seed)

    val_df = df[df["split"] == "val"]
    cal_df = df[(df["split"] == "cal") & (df["label"] == 0)]
    test_df = df[df["split"] == "test"]

    test_neg = test_df[test_df["label"] == 0]
    test_pos = test_df[test_df["label"] == 1]

    n_neg = len(test_neg)
    n_pos_target = int(n_neg * target_prev / (1 - target_prev))

    if n_pos_target > len(test_pos):
        n_pos_target = len(test_pos)
        log(f"  WARNING: not enough positives for {target_prev:.0%} prevalence, using {n_pos_target}")

    log(f"  Low-prevalence simulation: {n_neg} negatives + {n_pos_target} positives ({target_prev:.0%})")

    all_results = []
    for rep in range(n_repeats):
        pos_sample = test_pos.sample(n=n_pos_target, random_state=rng.randint(1e9))
        sim_test = pd.concat([test_neg, pos_sample])

        test_scores = sim_test[best_lm_col].values.astype(float)
        test_labels = sim_test["label"].values.astype(int)
        val_scores = val_df[best_lm_col].values.astype(float)
        val_labels = val_df["label"].values.astype(int)
        cal_scores = cal_df[best_lm_col].values.astype(float)

        test_scores = np.nan_to_num(test_scores, nan=0.5)
        val_scores = np.nan_to_num(val_scores, nan=0.5)
        cal_scores = np.nan_to_num(cal_scores, nan=0.5)

        for method_name, calibrate_fn in [
            ("Raw threshold", lambda vs=val_scores, vl=val_labels, ts=test_scores, a=alpha: calibrate_raw_threshold(vs, vl, ts, a)),
            ("Temp. scaling", lambda vs=val_scores, vl=val_labels, ts=test_scores, a=alpha: calibrate_temperature_scaling(vs, vl, ts, a)),
            ("Conformal", lambda cs=cal_scores, ts=test_scores, a=alpha: calibrate_conformal(cs, ts, a)),
        ]:
            predictions, _ = calibrate_fn()
            metrics = compute_metrics(predictions, test_labels, alpha)
            metrics["method"] = method_name
            metrics["rep"] = rep
            all_results.append(metrics)

    # Aggregate
    agg = pd.DataFrame(all_results)
    summary = agg.groupby("method").agg(
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        fpr_mean=("fpr", "mean"),
        fpr_std=("fpr", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        fp_mean=("fp", "mean"),
    ).round(4)

    log(f"\n  Low-prevalence results ({target_prev:.0%}):")
    log(summary.to_string())

    return summary


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="ConformalGuard calibration comparison")
    parser.add_argument("--scores-csv", required=True, help="CSV with scores from all LMs")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.01, help="Target FPR")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples")
    parser.add_argument("--multi-alpha", action="store_true",
                        help="Run at multiple alpha values for reliability plot")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("ConformalGuard: Calibration Method Comparison")
    log("=" * 60)

    # Load
    df = pd.read_csv(args.scores_csv)
    log(f"Loaded {len(df)} rows from {args.scores_csv}")
    log(f"Columns: {list(df.columns)}")
    log(f"Splits: {df['split'].value_counts().to_dict()}")
    log(f"Labels: {df['label'].value_counts().to_dict()}")

    # ---- Main comparison (Table 1) ----
    log(f"\n{'='*60}")
    log(f"TABLE 1: Calibration comparison at α={args.alpha}")
    log(f"{'='*60}")

    results = run_comparison(df, alpha=args.alpha, n_boot=args.n_boot)

    if not results:
        log("ERROR: No results produced. Check your score columns.")
        return

    # Save results
    results_flat = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ("ci", "info")}
        row["recall_std"] = r["ci"]["recall"]["std"]
        row["fpr_std"] = r["ci"]["fpr"]["std"]
        results_flat.append(row)

    results_df = pd.DataFrame(results_flat)
    results_df.to_csv(output_dir / "table1_calibration_comparison.csv", index=False)
    log(f"\nSaved to {output_dir}/table1_calibration_comparison.csv")

    # LaTeX
    latex = format_table_latex(results, args.alpha)
    with open(output_dir / "table1_calibration_comparison.tex", "w") as f:
        f.write(latex)
    log(f"Saved LaTeX to {output_dir}/table1_calibration_comparison.tex")

    # Print table
    log("\n" + "=" * 80)
    log("TABLE 1 (paper-ready)")
    log("=" * 80)
    for r in results:
        ci = r["ci"]
        viol = r["fpr_violation"]
        log(f"  {r['lm']:20s} | {r['calibration']:15s} | "
            f"Recall={r['recall']:.3f}±{ci['recall']['std']:.3f} | "
            f"FPR={r['fpr']:.4f} | Viol={viol:+.4f} | "
            f"Prec={r['precision']:.3f} | F1={r['f1']:.3f}")

    # ---- Multi-alpha reliability plot ----
    if args.multi_alpha:
        log(f"\n{'='*60}")
        log("CALIBRATION RELIABILITY PLOT (multiple α values)")
        log(f"{'='*60}")

        all_alpha_results = []
        for alpha in [0.005, 0.01, 0.02, 0.05, 0.10]:
            log(f"\n  α = {alpha}")
            alpha_results = run_comparison(df, alpha=alpha, n_boot=100)
            all_alpha_results.extend(alpha_results)

        plot_calibration_reliability(all_alpha_results,
                                     [0.005, 0.01, 0.02, 0.05, 0.10],
                                     output_dir)

    # ---- Low prevalence (Table 2) ----
    log(f"\n{'='*60}")
    log("TABLE 2: Low-prevalence evaluation (2%)")
    log(f"{'='*60}")

    # Select the best-performing LM column: most non-NaN rows, then highest mean score
    best_col: str | None = None
    best_valid = -1
    best_mean = -float("inf")
    for col in ["score_bert", "score_claude", "score_gpt"]:
        if col not in df.columns:
            continue
        valid_count = int(df[col].notna().sum())
        avg_score = float(df[col].mean(skipna=True)) if valid_count > 0 else -float("inf")
        if valid_count > best_valid or (valid_count == best_valid and avg_score > best_mean):
            best_col = col
            best_valid = valid_count
            best_mean = avg_score
    if best_col is None:
        raise ValueError("No score column (score_bert/score_claude/score_gpt) found in dataframe")

    lp_results = run_low_prevalence(df, best_col, alpha=args.alpha)
    lp_results.to_csv(output_dir / "table2_low_prevalence.csv")

    # ---- Summary ----
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")

    conformal_results = [r for r in results if r["calibration"] == "Conformal"]
    other_results = [r for r in results if r["calibration"] != "Conformal"]

    avg_conformal_viol = np.mean([r["fpr_violation"] for r in conformal_results])
    avg_other_viol = np.mean([r["fpr_violation"] for r in other_results])

    log(f"  Average FPR violation (conformal): {avg_conformal_viol:+.4f}")
    log(f"  Average FPR violation (other methods): {avg_other_viol:+.4f}")
    log(f"  Conformal advantage: {avg_other_viol - avg_conformal_viol:.4f}pp tighter")
    log("")
    log("  USE THESE NUMBERS IN THE ABSTRACT:")
    log(f"    [X] = {abs(avg_conformal_viol):.1f}pp (conformal violation)")
    log(f"    [Y] = {abs(avg_other_viol):.1f}pp (temp scaling average violation)")

    # Save full results as JSON for verification
    with open(output_dir / "metric_verification.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "ci"} for r in results],
                  f, indent=2, default=str)

    log(f"\nAll results saved to {output_dir}/")
    log("Done.")


if __name__ == "__main__":
    main()
