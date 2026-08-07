#!/usr/bin/env python3
"""Generate rebuttal analysis tables from existing ConformalGuard score files.

This script intentionally uses only the Python standard library so it can run
from the paper worktree without the experiment repo's virtual environment.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT.parent
if (ARTIFACT_ROOT / "results/bert_scores_v2.json").exists():
    RESULTS = ARTIFACT_ROOT / "results"
    SCORES = ARTIFACT_ROOT / "scores.csv"
    OUT = RESULTS / "camera_ready_analysis_tables.md"
else:
    experiment_root = os.environ.get("CONFORMALGUARD_EXPERIMENT_ROOT")
    if not experiment_root:
        raise RuntimeError(
            "Set CONFORMALGUARD_EXPERIMENT_ROOT when running outside the public artifact"
        )
    EXP = Path(experiment_root)
    RESULTS = EXP / "results"
    SCORES = EXP / "scores.csv"
    OUT = ROOT / "analysis-tables.md"


def as_float(value: str) -> float:
    try:
        if value == "" or value.lower() == "nan":
            return 0.5
        return float(value)
    except Exception:
        return 0.5


def read_scores() -> list[dict]:
    with SCORES.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["label_int"] = int(row["label"])
        for col in ("score_bert", "score_claude", "score_gpt"):
            row[col + "_float"] = as_float(row[col])
    return rows


def metrics(preds: list[int], labels: list[int]) -> dict:
    tp = sum(p == 1 and y == 1 for p, y in zip(preds, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(preds, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(preds, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(preds, labels))
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def conformal_preds(cal_scores: list[float], test_scores: list[float], alpha: float) -> list[int]:
    n = len(cal_scores)
    return [1 if (1 + sum(c >= s for c in cal_scores)) / (n + 1) <= alpha else 0 for s in test_scores]


def randomized_conformal_preds(
    cal_scores: list[float], test_scores: list[float], alpha: float, seed: int
) -> tuple[list[int], float]:
    """Randomized tie-breaking conformal p-values for high-score anomalies.

    For score s, deterministic conformal uses (1 + #{cal >= s}) / (n + 1).
    Randomized tie-breaking samples uniformly within the tied score block:
    p = (#{cal > s} + U * (#{cal = s} + 1)) / (n + 1).

    The returned dependency fraction counts examples whose decision can change
    with U, i.e. alpha falls inside the tied p-value interval.
    """
    rng = random.Random(seed)
    n = len(cal_scores)
    cal_sorted = sorted(cal_scores)
    preds: list[int] = []
    depends = 0
    for s in test_scores:
        left = bisect_left(cal_sorted, s)
        right = bisect_right(cal_sorted, s)
        greater = n - right
        equal = right - left
        low = greater / (n + 1)
        high = (greater + equal + 1) / (n + 1)
        if low < alpha < high:
            depends += 1
        p = (greater + rng.random() * (equal + 1)) / (n + 1)
        preds.append(1 if p <= alpha else 0)
    return preds, depends / max(len(test_scores), 1)


def raw_quantile_threshold(cal_scores_neg: list[float], alpha: float) -> float:
    vals = sorted(cal_scores_neg)
    if not vals:
        return float("inf")
    # Match numpy's default-ish linear quantile closely enough for threshold reporting.
    q = 1.0 - alpha
    idx = q * (len(vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vals[lo]
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def threshold_preds(scores: list[float], threshold: float) -> list[int]:
    return [1 if s >= threshold else 0 for s in scores]


def matched_threshold(test_scores: list[float], labels: list[int], target_fpr: float) -> tuple[float, dict]:
    # Oracle threshold on test labels: maximum recall subject to observed FPR <= target_fpr.
    best = None
    for t in sorted(set(test_scores), reverse=True):
        m = metrics(threshold_preds(test_scores, t), labels)
        if m["fpr"] <= target_fpr + 1e-12:
            best = (t, m)
        else:
            break
    if best is None:
        return max(test_scores) + 1e-9, metrics([0] * len(test_scores), labels)
    return best


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def logit(p: float) -> float:
    eps = 1e-7
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def fit_temperature(scores: list[float], labels: list[int]) -> float:
    def nll(temp: float) -> float:
        total = 0.0
        for s, y in zip(scores, labels):
            p = sigmoid(logit(s) / temp)
            p = min(max(p, 1e-9), 1 - 1e-9)
            total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        return total / max(len(scores), 1)

    # Ternary search on a unimodal-ish one-dimensional NLL.
    lo, hi = 0.01, 20.0
    for _ in range(120):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def transform_temperature(scores: list[float], temp: float) -> list[float]:
    return [sigmoid(logit(s) / temp) for s in scores]


def fit_isotonic_mapping(scores: list[float], labels: list[int]) -> dict[float, float]:
    # Pool-adjacent-violators for y sorted by score.
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    blocks: list[dict] = []
    for score, label in pairs:
        blocks.append({"sum": float(label), "n": 1, "scores": [score]})
        while len(blocks) >= 2:
            if blocks[-2]["sum"] / blocks[-2]["n"] <= blocks[-1]["sum"] / blocks[-1]["n"]:
                break
            b = blocks.pop()
            blocks[-1]["sum"] += b["sum"]
            blocks[-1]["n"] += b["n"]
            blocks[-1]["scores"].extend(b["scores"])
    mapping: dict[float, float] = {}
    for b in blocks:
        value = b["sum"] / b["n"]
        for score in b["scores"]:
            mapping[score] = value
    return mapping


def apply_isotonic(scores: list[float], mapping: dict[float, float]) -> list[float]:
    # Scores in val/cal/test are from the same finite grid for LLMs; for BERT, fallback
    # to nearest observed score if needed.
    keys = sorted(mapping)
    out = []
    for score in scores:
        if score in mapping:
            out.append(mapping[score])
            continue
        nearest = min(keys, key=lambda k: abs(k - score))
        out.append(mapping[nearest])
    return out


def fmt(x: float) -> str:
    return f"{x:.4f}"


def mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def std(xs: list[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs), 1))


def format_fraction_mean(values: list[Fraction]) -> str:
    exact_mean = sum(values, Fraction(0, 1)) / max(len(values), 1)
    decimal_mean = Decimal(exact_mean.numerator) / Decimal(exact_mean.denominator)
    return str(decimal_mean.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def percentile(xs: list[float], q: float) -> float:
    vals = sorted(xs)
    if not vals:
        return 0.0
    idx = q * (len(vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vals[lo]
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = read_scores()
    lms = [("BERT", "score_bert_float"), ("Claude Sonnet 4", "score_claude_float"), ("GPT-4o", "score_gpt_float")]
    test = [r for r in rows if r["split"] == "test"]
    val = [r for r in rows if r["split"] == "val"]
    cal_legit = [r for r in rows if r["split"] == "cal" and r["label_int"] == 0]

    sections: list[str] = []
    sections.append("# Camera-Ready Analysis Tables\n")
    sections.append("Generated from the bundled score artifacts.\n")

    sections.append("## Dataset Snapshot\n")
    split_counts = Counter((r["split"], r["label"]) for r in rows)
    source_counts = Counter(r["source"] for r in rows)
    sections.append(f"- Rows in `scores.csv`: {len(rows)}")
    sections.append(f"- Test positives / legitimate: {split_counts[('test', '1')]} / {split_counts[('test', '0')]}")
    sections.append(f"- Calibration legitimate examples: {split_counts[('cal', '0')]}")
    sections.append("- Top sources: " + ", ".join(f"{k}={v}" for k, v in source_counts.most_common(6)))
    sections.append("")

    # Existing main comparison.
    sections.append("## Table A — Main Calibration Comparison\n")
    with (RESULTS / "table1_calibration_comparison.csv").open(newline="") as f:
        main_rows = list(csv.DictReader(f))
    sections.append(md_table(
        ["LM", "Calibration", "Recall", "FPR", "Precision", "F1", "FPR viol."],
        [[r["lm"], r["calibration"], r["recall"], r["fpr"], r["precision"], r["f1"], r["fpr_violation"]] for r in main_rows],
    ))
    sections.append("")

    # Alpha sweep from current scores, not older full-pipeline conformal.
    alpha_rows = []
    for name, col in lms:
        cal_scores = [r[col] for r in cal_legit]
        test_scores = [r[col] for r in test]
        labels = [r["label_int"] for r in test]
        for alpha in [0.001, 0.005, 0.01, 0.05, 0.10]:
            m = metrics(conformal_preds(cal_scores, test_scores, alpha), labels)
            alpha_rows.append([name, str(alpha), fmt(m["recall"]), fmt(m["fpr"]), fmt(m["precision"]), fmt(m["f1"])])
    sections.append("## Table B — Conformal Threshold Sensitivity From Current Scores\n")
    sections.append(md_table(["LM", "alpha", "Recall", "FPR", "Precision", "F1"], alpha_rows))
    sections.append("")

    # Matched-FPR oracle threshold.
    matched_rows = []
    for name, col in lms:
        cal_scores = [r[col] for r in cal_legit]
        test_scores = [r[col] for r in test]
        labels = [r["label_int"] for r in test]
        conformal_m = metrics(conformal_preds(cal_scores, test_scores, 0.01), labels)
        threshold, raw_m = matched_threshold(test_scores, labels, conformal_m["fpr"])
        matched_rows.append([
            name,
            fmt(conformal_m["recall"]),
            fmt(conformal_m["fpr"]),
            f"{threshold:.6g}",
            fmt(raw_m["recall"]),
            fmt(raw_m["fpr"]),
            fmt(raw_m["f1"]),
        ])
    sections.append("## Table C — Oracle Matched-FPR Sanity Check\n")
    sections.append("Raw thresholding can match conformal decisions if it is allowed to choose the threshold using test labels. This table should be used to clarify that conformal's advantage is pre-deployment threshold validity, not dominance over an oracle threshold.\n")
    sections.append(md_table(["LM", "Conformal recall", "Conformal FPR", "Oracle raw threshold", "Oracle raw recall", "Oracle raw FPR", "Oracle raw F1"], matched_rows))
    sections.append("")

    # Balanced calibration ablation for temp/isotonic.
    balanced_rows = []
    for name, col in lms:
        val_neg = [r for r in val if r["label_int"] == 0]
        val_pos = [r for r in val if r["label_int"] == 1]
        n = min(len(val_neg), len(val_pos))
        balanced = val_neg[:n] + val_pos[:n]
        b_scores = [r[col] for r in balanced]
        b_labels = [r["label_int"] for r in balanced]
        test_scores = [r[col] for r in test]
        labels = [r["label_int"] for r in test]
        val_neg_scores = [r[col] for r in val_neg[:n]]

        temp = fit_temperature(b_scores, b_labels)
        b_temp_neg = transform_temperature(val_neg_scores, temp)
        test_temp = transform_temperature(test_scores, temp)
        temp_thresh = raw_quantile_threshold(b_temp_neg, 0.01)
        temp_m = metrics(threshold_preds(test_temp, temp_thresh), labels)

        iso = fit_isotonic_mapping(b_scores, b_labels)
        b_iso_neg = apply_isotonic(val_neg_scores, iso)
        test_iso = apply_isotonic(test_scores, iso)
        iso_thresh = raw_quantile_threshold(b_iso_neg, 0.01)
        iso_m = metrics(threshold_preds(test_iso, iso_thresh), labels)

        balanced_rows.append([name, "Temp. scaling balanced-val", fmt(temp_m["recall"]), fmt(temp_m["fpr"]), fmt(temp_m["precision"]), fmt(temp_m["f1"]), f"T={temp:.3f}"])
        balanced_rows.append([name, "Isotonic balanced-val", fmt(iso_m["recall"]), fmt(iso_m["fpr"]), fmt(iso_m["precision"]), fmt(iso_m["f1"]), f"threshold={iso_thresh:.3g}"])
    sections.append("## Table D — Balanced-Validation Calibration Ablation\n")
    sections.append("Temperature scaling and isotonic regression are fit on a balanced validation subset, then thresholded for alpha=0.01 using the legitimate half of that subset. This answers whether adding positives to the calibration fit fixes the LLM-score FPR issue.\n")
    sections.append(md_table(["LM", "Method", "Recall", "FPR", "Precision", "F1", "Fit detail"], balanced_rows))
    sections.append("")

    # Full-context/text-only summaries.
    summary_rows = []
    for path in sorted(RESULTS.glob("llm_baseline_*_summary.json")):
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict) or "test_metrics" not in data:
            continue
        model = data["model"].replace("claude-sonnet-4-20250514", "Claude Sonnet 4")
        mode = data["mode"]
        for op, m in data["test_metrics"].items():
            summary_rows.append([model, mode, op, fmt(m["recall"]), fmt(m["fpr"]), fmt(m["precision"]), fmt(m["f1"])])
    if summary_rows:
        sections.append("## Table E — Existing Text-Only vs Full-Context LLM Baselines\n")
        sections.append("These are not true few-shot or policy-prompt experiments, but they answer part of the richer-prompt concern using already-completed runs.\n")
        sections.append(md_table(["Model", "Prompt", "Operating point", "Recall", "FPR", "Precision", "F1"], summary_rows))
        sections.append("")

    # Source shift: calibrate on one legitimate source family, evaluate FPR on another.
    source_rows = []
    legit_test = [r for r in test if r["label_int"] == 0]
    source_groups = {
        "enron_ham": lambda r: r["source"].startswith("enron_") and r["source"].endswith("_ham"),
        "spamassassin_ham": lambda r: r["source"] == "spamassassin_ham",
    }
    for name, col in lms:
        for cal_name, cal_pred in source_groups.items():
            cal_subset = [r for r in rows if r["label_int"] == 0 and cal_pred(r) and r["split"] in {"cal", "val"}]
            if len(cal_subset) < 50:
                continue
            cal_scores = [r[col] for r in cal_subset]
            for test_name, test_pred in source_groups.items():
                test_subset = [r for r in legit_test if test_pred(r)]
                if len(test_subset) < 50:
                    continue
                preds = conformal_preds(cal_scores, [r[col] for r in test_subset], 0.01)
                fpr = sum(preds) / len(preds)
                source_rows.append([name, cal_name, str(len(cal_subset)), test_name, str(len(test_subset)), fmt(fpr), fmt(fpr - 0.01)])
    sections.append("## Table F — Legitimate-Source Shift Stress Test\n")
    sections.append("Calibration uses legitimate examples from one source family and evaluates FPR on legitimate examples from another. Positive-class recall is not measured here; this isolates exchangeability/FPR behavior on legitimate traffic.\n")
    sections.append(md_table(["LM", "Calibration source", "n_cal", "Test legitimate source", "n_test", "Observed FPR", "FPR - alpha"], source_rows))
    sections.append("")

    # Low-prevalence workload, scaled correctly to a 10K-email inbox.
    low_prev_rows = []
    bert_col = "score_bert_float"
    labels = [r["label_int"] for r in test]
    test_scores = [r[bert_col] for r in test]
    val_neg_scores = [r[bert_col] for r in val if r["label_int"] == 0]
    cal_scores = [r[bert_col] for r in cal_legit]
    raw_preds = threshold_preds(test_scores, raw_quantile_threshold(val_neg_scores, 0.01))
    temp_preds = raw_preds[:]  # Temperature scaling is decision-identical for BERT in the released table.
    conf_preds = conformal_preds(cal_scores, test_scores, 0.01)
    method_preds = [("Raw threshold", raw_preds), ("Temp. scaling", temp_preds), ("Conformal", conf_preds)]

    legit_idx = [i for i, y in enumerate(labels) if y == 0]
    n_legit = len(legit_idx)
    scale = 9800 / n_legit  # 2% phishing prevalence => 9,800 legitimate emails per 10,000.
    rng = random.Random(20260527)
    boot_fp: dict[str, list[float]] = {name: [] for name, _ in method_preds}
    raw_only: list[float] = []
    for _ in range(5000):
        sample = [rng.choice(legit_idx) for _ in legit_idx]
        raw_fp = sum(raw_preds[i] for i in sample) * scale
        conf_fp = sum(conf_preds[i] for i in sample) * scale
        boot_fp["Raw threshold"].append(raw_fp)
        boot_fp["Temp. scaling"].append(raw_fp)
        boot_fp["Conformal"].append(conf_fp)
        raw_only.append(raw_fp - conf_fp)

    for name, preds in method_preds:
        m = metrics(preds, labels)
        fp_per_10k = m["fpr"] * 9800
        ci = (percentile(boot_fp[name], 0.025), percentile(boot_fp[name], 0.975))
        tp_per_10k = m["recall"] * 200
        precision_2pct = tp_per_10k / (tp_per_10k + fp_per_10k)
        low_prev_rows.append([
            name,
            fmt(m["recall"]),
            fmt(m["fpr"]),
            fmt(precision_2pct),
            f"{fp_per_10k:.0f} [{ci[0]:.0f}, {ci[1]:.0f}]",
        ])
    diff_ci = (percentile(raw_only, 0.025), percentile(raw_only, 0.975))
    sections.append("## Table G — Low-Prevalence Workload With Correct Per-10K Scaling\n")
    sections.append("The earlier `fp_mean` field in `table2_low_prevalence.csv` is the raw false-positive count over 3,911 legitimate test emails. This table rescales FPR to a 10,000-email inbox at 2% phishing prevalence, i.e. 9,800 legitimate emails. Brackets are paired bootstrap 95% intervals over legitimate test emails. The raw-minus-conformal reduction is {:.0f} FP per 10K emails, 95% CI [{:.0f}, {:.0f}].\n".format(
        (metrics(raw_preds, labels)["fpr"] - metrics(conf_preds, labels)["fpr"]) * 9800,
        diff_ci[0],
        diff_ci[1],
    ))
    sections.append(md_table(["Calibration", "Recall", "FPR", "Precision at 2%", "FP per 10K emails [95% CI]"], low_prev_rows))
    sections.append("")

    # Randomized conformal baseline / auditability diagnostic.
    rand_rows = []
    for name, col in lms:
        cal_scores = [r[col] for r in cal_legit]
        test_scores = [r[col] for r in test]
        labels = [r["label_int"] for r in test]
        for alpha in [0.001, 0.005, 0.01, 0.05, 0.10]:
            metric_lists: dict[str, list[float]] = defaultdict(list)
            metric_fractions: dict[str, list[Fraction]] = defaultdict(list)
            dep_fracs: list[float] = []
            first_preds = None
            disagreements: list[float] = []
            for seed in range(100):
                preds, dep_frac = randomized_conformal_preds(cal_scores, test_scores, alpha, seed)
                if first_preds is None:
                    first_preds = preds
                else:
                    disagreements.append(sum(a != b for a, b in zip(first_preds, preds)) / len(preds))
                m = metrics(preds, labels)
                dep_fracs.append(dep_frac)
                for key in ("recall", "fpr", "precision", "f1"):
                    metric_lists[key].append(m[key])
                metric_fractions["recall"].append(Fraction(m["tp"], m["tp"] + m["fn"]))
                metric_fractions["fpr"].append(Fraction(m["fp"], m["fp"] + m["tn"]))
                metric_fractions["precision"].append(Fraction(m["tp"], max(m["tp"] + m["fp"], 1)))
                metric_fractions["f1"].append(Fraction(2 * m["tp"], max(2 * m["tp"] + m["fp"] + m["fn"], 1)))
            rand_rows.append([
                name,
                str(alpha),
                f"{format_fraction_mean(metric_fractions['recall'])} ± {std(metric_lists['recall']):.3f}",
                f"{format_fraction_mean(metric_fractions['fpr'])} ± {std(metric_lists['fpr']):.3f}",
                format_fraction_mean(metric_fractions["precision"]),
                format_fraction_mean(metric_fractions["f1"]),
                f"{mean(dep_fracs):.1%}",
                f"{mean(disagreements):.1%}",
            ])
    sections.append("## Table H — Randomized Conformal Tie-Breaking Baseline\n")
    sections.append("Randomized conformal tie-breaking can split large tied score blocks, but this makes action assignment stochastic. `Decision depends on U` is the fraction of test examples whose p-value interval straddles alpha; `seed disagreement` is the average fraction of test examples whose decision differs from seed 0 across 99 other random seeds. This table directly answers whether randomized tie-breaking is a free fix for coarse LLM scores.\n")
    sections.append(md_table(
        ["LM", "alpha", "Recall", "FPR", "Precision", "F1", "Decision depends on U", "Seed disagreement"],
        rand_rows,
    ))
    sections.append("")

    OUT.write_text("\n".join(sections) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
