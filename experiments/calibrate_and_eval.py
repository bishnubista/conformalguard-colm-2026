#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "transformers>=4.40",
#   "torch>=2.2",
#   "scikit-learn>=1.3",
#   "scipy>=1.12",
#   "numpy>=1.26",
#   "tqdm>=4.65",
# ]
# ///
"""
ConformalGuard calibration and evaluation pipeline.

Stages:
  1. BERT inference  -- run fine-tuned model on cal + test to get s_text scores
  2. Scorer 2 (auth) -- Random Forest on header features, trained on train split
  3. Scorer 3 (URL)  -- Random Forest on URL features, trained on train split
  4. Weight learning -- optimize w1, w2, w3 on val split
  5. Conformal cal   -- compute q_hat at each alpha on cal set (legit only)
  6. Evaluation      -- recall, FPR, precision, F1 + 95% bootstrap CIs on test
  7. Ablation        -- remove each scorer and re-calibrate/re-evaluate
  8. Cal size sweep  -- subsample cal set at various n, measure FPR stability

Usage:
    uv run experiments/calibrate_and_eval.py
    uv run experiments/calibrate_and_eval.py --data-dir data/splits --model-dir models/bert_scorer
    uv run experiments/calibrate_and_eval.py --skip-cache  # force re-run BERT inference

Results saved to results/conformal_results.json and printed as formatted tables.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Target alpha levels for conformal calibration
ALPHAS = [0.001, 0.005, 0.01, 0.05, 0.10]

# Calibration set sizes for sensitivity sweep
CAL_SIZES = [100, 500, 1000, 2000, 3000]

# Bootstrap resamples for confidence intervals
N_BOOTSTRAP = 1000

# Random seed
SEED = 42
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        log.info("Inference device: CUDA (%s)", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        log.info("Inference device: Apple MPS")
    else:
        device = "cpu"
        log.info("Inference device: CPU")
    return device


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits(data_dir: Path) -> dict:
    """Load all four splits from parquet files."""
    splits = {}
    for name in ("train", "val", "cal", "test"):
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            log.error("Split not found: %s", path)
            log.error("Run: uv run scripts/parse_and_split.py")
            sys.exit(1)
        df = pd.read_parquet(path)
        # Combined text input for BERT
        df["text"] = (
            df["subject"].fillna("").str.strip()
            + " [SEP] "
            + df["body_text"].fillna("").str.strip()
        )
        splits[name] = df
        n_legit = (df["label"] == 0).sum()
        n_phish = (df["label"] == 1).sum()
        log.info("  %s: %d emails (%d legit, %d phishing)", name, len(df), n_legit, n_phish)
    return splits


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add any derived columns needed by the scorers."""
    df = df.copy()
    df["has_sender_ip"] = (df["sender_ip"].fillna("") != "").astype(int)
    df["received_hops"] = df["received_hops"].clip(0, 30).fillna(0).astype(float)
    for col in ["url_count", "has_ip_url", "has_shortener", "has_href_mismatch",
                "from_reply_to_mismatch", "from_return_path_mismatch"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(float)
    return df


# ---------------------------------------------------------------------------
# Stage 1: BERT inference
# ---------------------------------------------------------------------------

AUTH_FEATURES = [
    "from_reply_to_mismatch",
    "from_return_path_mismatch",
    "received_hops",
    "has_sender_ip",
]

URL_FEATURES = [
    "url_count",
    "has_ip_url",
    "has_shortener",
    "has_href_mismatch",
]


def bert_inference_batch(
    df: pd.DataFrame,
    model_dir: Path,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    """Return P(phishing) for every row using the fine-tuned BERT model."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    log.info("  Loading BERT from: %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    texts = df["text"].tolist()
    all_probs: list[float] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="  BERT inference", unit="batch"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits  # (B, 2)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())

    return np.array(all_probs, dtype=np.float64)


def run_bert_inference_all(
    splits: dict,
    model_dir: Path,
    cache_dir: Path,
    batch_size: int,
    max_length: int,
    skip_cache: bool,
) -> dict:
    """Run BERT inference on val, cal, test splits; cache arrays to disk."""
    device = detect_device()
    scores: dict[str, np.ndarray] = {}

    for name in ("val", "cal", "test"):
        cache_path = cache_dir / f"s_text_{name}.npy"
        if not skip_cache and cache_path.exists():
            # WARNING: Cache is keyed only by split name — if the underlying parquet
            # data or checkpoint changes, stale scores will be reused silently.
            # Pass --skip-cache to force re-inference after checkpoint/data updates.
            log.info("  Loading cached s_text for %s: %s", name, cache_path)
            log.warning("  (stale cache risk) Use --skip-cache if checkpoint or data changed")
            scores[name] = np.load(str(cache_path))
        else:
            log.info("  Running BERT inference on %s split (%d emails)...",
                     name, len(splits[name]))
            scores[name] = bert_inference_batch(
                splits[name], model_dir, batch_size, max_length, device
            )
            np.save(str(cache_path), scores[name])
            log.info("  Cached to %s", cache_path)

    return scores


# ---------------------------------------------------------------------------
# Stage 2 & 3: Header + URL Random Forest scorers
# ---------------------------------------------------------------------------

def train_rf_scorer(train_df: pd.DataFrame, features: list, name: str):
    """Train a Random Forest scorer and return the fitted classifier."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    X = train_df[features].values
    y = train_df["label"].astype(int).values

    log.info("  Training %s Random Forest on %d emails, %d features...",
             name, len(train_df), len(features))
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X, y)

    train_proba = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, train_proba)
    log.info("  %s RF train ROC-AUC: %.4f", name, auc)
    for feat, imp in sorted(zip(features, clf.feature_importances_), key=lambda t: -t[1]):
        log.info("    %-30s %.4f", feat, imp)

    return clf


def score_rf(clf, df: pd.DataFrame, features: list) -> np.ndarray:
    """Return P(phishing) from a trained Random Forest."""
    return clf.predict_proba(df[features].values)[:, 1]


# ---------------------------------------------------------------------------
# Stage 4: Weight learning via global optimization
# ---------------------------------------------------------------------------

def learn_weights(
    s_text_val: np.ndarray,
    s_auth_val: np.ndarray,
    s_url_val: np.ndarray,
    labels_val: np.ndarray,
    target_fpr: float = 0.05,
) -> tuple:
    """
    Learn w1, w2, w3 on the validation set.

    Objective: maximize phishing recall subject to FPR <= target_fpr on val.
    Aggregation: f(x) = max(w1*s_text, w2*s_auth, w3*s_url)

    Strategy: differential evolution (global) followed by L-BFGS-B (local polish).
    """
    from scipy.optimize import differential_evolution, minimize

    def objective(w):
        w1, w2, w3 = np.clip(w, 0.0, None)
        total = w1 + w2 + w3
        if total < 1e-8:
            return 1.0
        # Normalize so weights sum to 3.0 (preserves scale comparable to raw scores)
        w1, w2, w3 = 3.0 * w1 / total, 3.0 * w2 / total, 3.0 * w3 / total

        agg = np.maximum.reduce([w1 * s_text_val, w2 * s_auth_val, w3 * s_url_val])
        legit_scores = agg[labels_val == 0]
        phish_scores = agg[labels_val == 1]

        if len(legit_scores) == 0 or len(phish_scores) == 0:
            return 1.0

        threshold = np.quantile(legit_scores, 1.0 - target_fpr)
        recall = (phish_scores > threshold).mean()
        fpr = (legit_scores > threshold).mean()
        # Penalize FPR violations strongly
        penalty = max(0.0, fpr - target_fpr) * 100.0
        return float(-recall + penalty)

    log.info("  Optimizing weights (target val FPR <= %.3f)...", target_fpr)
    bounds = [(0.01, 3.0), (0.01, 3.0), (0.01, 3.0)]

    de_result = differential_evolution(
        objective, bounds, seed=SEED, maxiter=500, popsize=15, tol=1e-6, workers=1
    )
    polish = minimize(
        objective, de_result.x, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    best = polish.x if polish.success else de_result.x

    w1, w2, w3 = np.clip(best, 0.0, None)
    total = w1 + w2 + w3
    w1, w2, w3 = 3.0 * w1 / total, 3.0 * w2 / total, 3.0 * w3 / total

    log.info("  Learned weights: w1(text)=%.4f, w2(auth)=%.4f, w3(url)=%.4f", w1, w2, w3)
    return float(w1), float(w2), float(w3)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    s_text: np.ndarray,
    s_auth: np.ndarray,
    s_url: np.ndarray,
    w1: float,
    w2: float,
    w3: float,
) -> np.ndarray:
    """f(x) = max(w1*s_text, w2*s_auth, w3*s_url)"""
    return np.maximum.reduce([w1 * s_text, w2 * s_auth, w3 * s_url])


# ---------------------------------------------------------------------------
# Stage 5: Conformal calibration
# ---------------------------------------------------------------------------

def conformal_thresholds(cal_scores: np.ndarray, alphas: list) -> dict:
    """
    Compute q_hat = Quantile_{1-alpha}(cal_scores) for each alpha.

    Uses the conservative finite-sample level ceil((1-alpha)(n+1))/n so that
    P(legitimate email flagged) <= alpha under exchangeability.
    """
    n = len(cal_scores)
    if n == 0:
        raise ValueError(
            "Calibration set is empty — the defensive filter removed all legitimate emails. "
            "Check your data splits before running calibration."
        )
    log.info("  Calibration set: %d legitimate emails", n)
    thresholds = {}
    for alpha in alphas:
        level = min(np.ceil((1.0 - alpha) * (n + 1)) / n, 1.0)
        q_hat = float(np.quantile(cal_scores, level, method="higher"))
        thresholds[alpha] = q_hat
        log.info("    alpha=%.3f: q_hat=%.6f  (quantile level=%.4f)", alpha, q_hat, level)
    return thresholds


# ---------------------------------------------------------------------------
# Stage 6: Evaluation + bootstrap CIs
# ---------------------------------------------------------------------------

def compute_binary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """
    Threshold scores and compute recall, FPR (legitimate intervention rate),
    precision, and F1. Also returns raw confusion matrix counts.
    """
    predicted = (scores > threshold).astype(int)

    legit_mask = labels == 0
    phish_mask = labels == 1

    tp = int(((predicted == 1) & phish_mask).sum())
    fn = int(((predicted == 0) & phish_mask).sum())
    fp = int(((predicted == 1) & legit_mask).sum())
    tn = int(((predicted == 0) & legit_mask).sum())

    n_phish = int(phish_mask.sum())
    n_legit = int(legit_mask.sum())

    recall    = tp / n_phish if n_phish > 0 else 0.0
    fpr       = fp / n_legit if n_legit > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "recall": float(recall),
        "fpr": float(fpr),
        "precision": float(precision),
        "f1": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_phish": n_phish, "n_legit": n_legit,
    }


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    n_boot: int = N_BOOTSTRAP,
    ci_level: float = 0.95,
) -> dict:
    """
    Percentile-bootstrap 95% confidence intervals for recall, FPR, precision, F1.
    """
    n = len(scores)
    boot_metrics: dict[str, list] = {"recall": [], "fpr": [], "precision": [], "f1": []}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_binary_metrics(scores[idx], labels[idx], threshold)
        for key in boot_metrics:
            boot_metrics[key].append(m[key])

    lo = (1.0 - ci_level) / 2.0
    hi = 1.0 - lo
    cis: dict[str, float] = {}
    for key, vals in boot_metrics.items():
        arr = np.array(vals)
        cis[f"{key}_ci_lo"] = float(np.quantile(arr, lo))
        cis[f"{key}_ci_hi"] = float(np.quantile(arr, hi))
    return cis


def full_test_evaluation(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    thresholds: dict,
    pipeline_label: str = "full_pipeline",
) -> list:
    """Evaluate at every alpha with bootstrap CIs. Returns list of result dicts."""
    results = []
    for alpha in ALPHAS:
        threshold = thresholds[alpha]
        point = compute_binary_metrics(test_scores, test_labels, threshold)
        cis = bootstrap_ci(test_scores, test_labels, threshold)
        row = {
            "pipeline": pipeline_label,
            "alpha": alpha,
            "threshold": round(threshold, 6),
            **point,
            **cis,
        }
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Stage 7: Ablation
# ---------------------------------------------------------------------------

def run_ablation(
    splits: dict,
    bert_scores: dict,
    auth_clf,
    url_clf,
    weights: tuple,
) -> list:
    """
    Remove each scorer one at a time (set its weight to 0), re-calibrate on
    the cal split, and re-evaluate on the test split.
    """
    w1, w2, w3 = weights
    cal_labels  = splits["cal"]["label"].values
    test_labels = splits["test"]["label"].values

    s_text_cal  = bert_scores["cal"]
    s_auth_cal  = score_rf(auth_clf, splits["cal"],  AUTH_FEATURES)
    s_url_cal   = score_rf(url_clf,  splits["cal"],  URL_FEATURES)
    s_text_test = bert_scores["test"]
    s_auth_test = score_rf(auth_clf, splits["test"], AUTH_FEATURES)
    s_url_test  = score_rf(url_clf,  splits["test"], URL_FEATURES)

    configs = [
        ("no_text",   0.0,  w2,  w3,  "Without Scorer 1 (BERT text)"),
        ("no_auth",   w1,  0.0,  w3,  "Without Scorer 2 (header auth)"),
        ("no_url",    w1,  w2,  0.0,  "Without Scorer 3 (URL features)"),
        ("text_only", w1,  0.0, 0.0,  "Scorer 1 only (BERT text)"),
        ("auth_only", 0.0, w2,  0.0,  "Scorer 2 only (header auth)"),
        ("url_only",  0.0, 0.0, w3,  "Scorer 3 only (URL features)"),
    ]

    all_results = []
    for cfg_id, aw1, aw2, aw3, desc in configs:
        log.info("  Ablation: %s", desc)
        cal_agg  = aggregate(s_text_cal,  s_auth_cal,  s_url_cal,  aw1, aw2, aw3)
        test_agg = aggregate(s_text_test, s_auth_test, s_url_test, aw1, aw2, aw3)

        cal_legit_scores = cal_agg[cal_labels == 0]
        thresholds = conformal_thresholds(cal_legit_scores, ALPHAS)
        rows = full_test_evaluation(test_agg, test_labels, thresholds, pipeline_label=cfg_id)
        for r in rows:
            r["ablation_desc"] = desc
        all_results.extend(rows)

    return all_results


# ---------------------------------------------------------------------------
# Stage 8: Calibration set size sensitivity sweep
# ---------------------------------------------------------------------------

def run_cal_size_sweep(
    cal_legit_scores: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    cal_sizes: list,
    n_repeats: int = 10,
) -> list:
    """
    For each calibration set size n, subsample the cal set n_repeats times,
    compute thresholds, evaluate on test, and report mean +/- std of FPR and recall.
    """
    available = len(cal_legit_scores)
    results = []

    for n in cal_sizes:
        if n > available:
            log.info("  Cal size %d > available %d, skipping", n, available)
            continue

        log.info("  Cal size sweep: n=%d (%d repeats)", n, n_repeats)
        per_alpha: dict = {a: {"recall": [], "fpr": []} for a in ALPHAS}

        for rep in range(n_repeats):
            rep_rng = np.random.default_rng(SEED + rep)
            idx = rep_rng.choice(available, size=n, replace=False)
            sub_scores = cal_legit_scores[idx]
            thresholds = conformal_thresholds(sub_scores, ALPHAS)

            for alpha in ALPHAS:
                m = compute_binary_metrics(test_scores, test_labels, thresholds[alpha])
                per_alpha[alpha]["recall"].append(m["recall"])
                per_alpha[alpha]["fpr"].append(m["fpr"])

        for alpha in ALPHAS:
            recalls = np.array(per_alpha[alpha]["recall"])
            fprs    = np.array(per_alpha[alpha]["fpr"])
            results.append({
                "cal_n": n,
                "alpha": alpha,
                "recall_mean": float(recalls.mean()),
                "recall_std":  float(recalls.std()),
                "fpr_mean":    float(fprs.mean()),
                "fpr_std":     float(fprs.std()),
                # True if FPR <= alpha in ALL repeats
                "fpr_controlled": bool((fprs <= alpha + 1e-9).all()),
            })

    return results


# ---------------------------------------------------------------------------
# Pretty-printing (paper table format)
# ---------------------------------------------------------------------------

def print_main_table(results: list, title: str = "Main Results") -> None:
    """Print evaluation results as a formatted table (Table 2/3 style)."""
    print()
    print("=" * 106)
    print(f"  {title}")
    print("=" * 106)
    hdr = (
        f"{'Pipeline':<22} {'Alpha':>6} {'Threshold':>10} "
        f"{'Recall':>8} {'[95% CI]':>16} "
        f"{'Legit IR':>9} {'[95% CI]':>16} "
        f"{'Prec':>7} {'F1':>7}"
    )
    print(hdr)
    print("-" * 106)
    for r in results:
        print(
            f"{r['pipeline']:<22} {r['alpha']:>6.3f} {r['threshold']:>10.6f} "
            f"{r['recall']:>8.4f} [{r.get('recall_ci_lo', float('nan')):.4f},{r.get('recall_ci_hi', float('nan')):.4f}] "
            f"{r['fpr']:>9.4f} [{r.get('fpr_ci_lo', float('nan')):.4f},{r.get('fpr_ci_hi', float('nan')):.4f}] "
            f"{r['precision']:>7.4f} {r['f1']:>7.4f}"
        )
    print("=" * 106)
    print("  Note: Legit IR = Legitimate Intervention Rate = empirical FPR on test set")
    print()


def print_confusion_matrix(row: dict) -> None:
    tp, tn, fp, fn = row["tp"], row["tn"], row["fp"], row["fn"]
    alpha = row["alpha"]
    thresh = row["threshold"]
    print(f"\n  Confusion Matrix (alpha={alpha:.3f}, threshold={thresh:.6f})")
    print(f"  {'':16s} {'Pred: Legit':>13} {'Pred: Phish':>13}")
    print(f"  {'True: Legit':16s} {tn:>13,} {fp:>13,}")
    print(f"  {'True: Phish':16s} {fn:>13,} {tp:>13,}")
    print()


def print_ablation_table(results: list) -> None:
    print()
    print("=" * 82)
    print("  Ablation Study")
    print("=" * 82)
    print(f"{'Config':<22} {'Alpha':>6} {'Recall':>8} {'Legit IR':>9} {'Prec':>7} {'F1':>7}")
    print("-" * 82)
    prev_alpha = None
    for r in sorted(results, key=lambda x: (x["alpha"], x["pipeline"])):
        if r["alpha"] != prev_alpha:
            if prev_alpha is not None:
                print()
            prev_alpha = r["alpha"]
        print(
            f"{r['pipeline']:<22} {r['alpha']:>6.3f} "
            f"{r['recall']:>8.4f} {r['fpr']:>9.4f} "
            f"{r['precision']:>7.4f} {r['f1']:>7.4f}"
        )
    print("=" * 82)
    print()


def print_cal_sweep_table(results: list) -> None:
    print()
    print("=" * 78)
    print("  Calibration Set Size Sensitivity")
    print("=" * 78)
    print(
        f"{'Cal n':>8} {'Alpha':>7}  "
        f"{'Recall (mean+/-std)':>22}  {'Legit IR (mean+/-std)':>22}  {'Controlled':>10}"
    )
    print("-" * 78)
    prev_n = None
    for r in sorted(results, key=lambda x: (x["cal_n"], x["alpha"])):
        if r["cal_n"] != prev_n:
            if prev_n is not None:
                print()
            prev_n = r["cal_n"]
        controlled = "YES" if r["fpr_controlled"] else "NO "
        print(
            f"{r['cal_n']:>8} {r['alpha']:>7.3f}  "
            f"{r['recall_mean']:.4f} +/- {r['recall_std']:.4f}       "
            f"{r['fpr_mean']:.4f} +/- {r['fpr_std']:.4f}       "
            f"{controlled}"
        )
    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args):
    data_dir    = Path(args.data_dir)
    model_dir   = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  ConformalGuard: Calibration & Evaluation Pipeline")
    log.info("=" * 60)

    # -- 1. Load splits -------------------------------------------------------
    log.info("\n[1/8] Loading data splits...")
    splits = load_splits(data_dir)
    for name in splits:
        splits[name] = add_derived_features(splits[name])

    # -- 2. BERT inference (Scorer 1) -----------------------------------------
    log.info("\n[2/8] BERT inference (Scorer 1: text)...")
    if not model_dir.is_dir():
        log.error("BERT model not found: %s", model_dir)
        log.error("Run: uv run experiments/train_bert.py")
        sys.exit(1)

    # Create score cache only after confirming model_dir exists (guard above fires first)
    cache_dir = model_dir / "score_cache"
    cache_dir.mkdir(exist_ok=True)

    bert_scores = run_bert_inference_all(
        splits, model_dir, cache_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        skip_cache=args.skip_cache,
    )
    for split_name in ("val", "cal", "test"):
        s = bert_scores[split_name]
        log.info("  s_text %-5s | mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                 split_name, s.mean(), s.std(), s.min(), s.max())

    # -- 3. Scorer 2: header auth Random Forest --------------------------------
    log.info("\n[3/8] Scorer 2: header authentication Random Forest...")
    auth_clf = train_rf_scorer(splits["train"], AUTH_FEATURES, "Auth")

    # -- 4. Scorer 3: URL Random Forest ----------------------------------------
    log.info("\n[4/8] Scorer 3: URL feature Random Forest...")
    url_clf = train_rf_scorer(splits["train"], URL_FEATURES, "URL")

    # Compute raw scorer scores on val/cal/test
    log.info("\n  Computing auth + URL scores on val, cal, test...")
    s_auth: dict[str, np.ndarray] = {
        split_name: score_rf(auth_clf, splits[split_name], AUTH_FEATURES)
        for split_name in ("val", "cal", "test")
    }
    s_url: dict[str, np.ndarray] = {
        split_name: score_rf(url_clf, splits[split_name], URL_FEATURES)
        for split_name in ("val", "cal", "test")
    }

    # -- 5. Weight learning on val --------------------------------------------
    log.info("\n[5/8] Learning aggregation weights on validation set...")
    val_labels = splits["val"]["label"].values
    w1, w2, w3 = learn_weights(
        bert_scores["val"], s_auth["val"], s_url["val"],
        val_labels,
        target_fpr=args.weight_fpr_target,
    )
    weights = (w1, w2, w3)

    # -- 5b. Aggregate scores -------------------------------------------------
    f_cal  = aggregate(bert_scores["cal"],  s_auth["cal"],  s_url["cal"],  w1, w2, w3)
    f_test = aggregate(bert_scores["test"], s_auth["test"], s_url["test"], w1, w2, w3)

    cal_labels  = splits["cal"]["label"].values
    test_labels = splits["test"]["label"].values

    # Cal set should be all legit; filter defensively
    cal_legit_mask = cal_labels == 0
    n_phish_in_cal = int((~cal_legit_mask).sum())
    if n_phish_in_cal > 0:
        log.warning("Cal set contains %d non-legit emails -- excluded from calibration",
                    n_phish_in_cal)
    cal_legit_scores = f_cal[cal_legit_mask]

    # -- 6. Conformal calibration ---------------------------------------------
    log.info("\n[6/8] Conformal calibration (cal set)...")
    thresholds = conformal_thresholds(cal_legit_scores, ALPHAS)

    # -- 7. Main test evaluation ----------------------------------------------
    log.info("\n[7/8] Evaluating on test set (full pipeline)...")
    main_results = full_test_evaluation(f_test, test_labels, thresholds, "full_pipeline")

    print_main_table(main_results, title="Table 2: ConformalGuard -- Clean Test Set")

    # Confusion matrices at two representative thresholds
    for target_alpha in [0.01, 0.05]:
        row = next(r for r in main_results if abs(r["alpha"] - target_alpha) < 1e-9)
        print_confusion_matrix(row)

    # -- 8a. Ablation study ---------------------------------------------------
    log.info("\n[8a/8] Running ablation study...")
    ablation_results = run_ablation(splits, bert_scores, auth_clf, url_clf, weights)
    print_ablation_table(ablation_results)

    # -- 8b. Calibration size sensitivity sweep --------------------------------
    log.info("\n[8b/8] Running calibration set size sensitivity sweep...")
    cal_sweep_results = run_cal_size_sweep(
        cal_legit_scores, f_test, test_labels, CAL_SIZES, n_repeats=10
    )
    print_cal_sweep_table(cal_sweep_results)

    # -- Save all results to JSON ---------------------------------------------
    output = {
        "config": {
            "data_dir":    str(data_dir),
            "model_dir":   str(model_dir),
            "alphas":      ALPHAS,
            "cal_sizes":   CAL_SIZES,
            "n_bootstrap": N_BOOTSTRAP,
            "weights": {"w1_text": w1, "w2_auth": w2, "w3_url": w3},
            "conformal_thresholds": {str(k): v for k, v in thresholds.items()},
            "cal_n":       int(cal_legit_scores.shape[0]),
            "test_n":      int(test_labels.shape[0]),
            "test_n_phish": int((test_labels == 1).sum()),
            "test_n_legit": int((test_labels == 0).sum()),
        },
        "main_results":    main_results,
        "ablation_results": ablation_results,
        "cal_size_sweep":   cal_sweep_results,
    }

    out_path = results_dir / "conformal_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Results saved: %s", out_path)

    # -- Final summary --------------------------------------------------------
    row_01 = next(r for r in main_results if abs(r["alpha"] - 0.01) < 1e-9)
    print()
    print("=" * 60)
    print("  Summary at alpha=0.01 (target FPR <= 1%)")
    print("=" * 60)
    print(f"  Recall (phishing detection):  {row_01['recall']:.4f}  "
          f"[{row_01['recall_ci_lo']:.4f}, {row_01['recall_ci_hi']:.4f}]")
    print(f"  Legitimate intervention rate: {row_01['fpr']:.4f}  "
          f"[{row_01['fpr_ci_lo']:.4f}, {row_01['fpr_ci_hi']:.4f}]")
    print(f"  Conformal guarantee holds:    {row_01['fpr'] <= 0.01}")
    print(f"  Precision:                    {row_01['precision']:.4f}")
    print(f"  F1:                           {row_01['f1']:.4f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ConformalGuard: conformal calibration and evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="data/splits",
        help="Directory containing train/val/cal/test parquet splits",
    )
    parser.add_argument(
        "--model-dir", default="models/bert_scorer",
        help="Path to fine-tuned BERT model (output of train_bert.py)",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Output directory for JSON results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for BERT inference",
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Max token length for BERT tokenization",
    )
    parser.add_argument(
        "--weight-fpr-target", type=float, default=0.05,
        help="Target FPR on val set used when optimizing aggregation weights",
    )
    parser.add_argument(
        "--skip-cache", action="store_true", default=False,
        help="Force re-run BERT inference even if cached .npy score files exist",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
