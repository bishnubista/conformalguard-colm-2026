#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "pyarrow>=14.0"]
# ///
"""
Assemble BERT, Claude, and GPT scores into a single scores.csv for the
calibration comparison experiment.

Usage:
    uv run experiments/assemble_scores.py

Input:
    results/bert_scores_v2.json
    results/llm_scores_claude_v2.json
    results/llm_scores_gpt_v2.json
    data/splits_v2/*.parquet (for split/label ground truth)

Output:
    scores.csv with columns: split, label, source, score_bert, score_claude, score_gpt
"""

import json
from pathlib import Path

import pandas as pd


def load_json_scores(path, score_key="score"):
    """Load scores from a JSON results file."""
    with open(path) as f:
        data = json.load(f)
    # Only keep non-errored results
    records = []
    for r in data:
        if not r.get("error"):
            records.append({
                "email_id": r["email_id"],
                score_key: r.get("score", r.get("score_bert", 0.5)),
            })
    return pd.DataFrame(records)


def main():
    # Load split ground truth
    splits_dir = Path("data/splits_v2")
    frames = []
    for split_name in ["val", "cal", "test"]:
        df = pd.read_parquet(splits_dir / f"{split_name}.parquet")
        df["split"] = split_name
        frames.append(df[["email_id", "split", "label", "source"]])
    ground_truth = pd.concat(frames, ignore_index=True)
    print(f"Ground truth: {len(ground_truth)} emails across {ground_truth['split'].value_counts().to_dict()}")

    # Load BERT scores
    bert_path = Path("results/bert_scores_v2.json")
    if bert_path.exists():
        bert_df = load_json_scores(bert_path, "score_bert")
        print(f"BERT scores: {len(bert_df)}")
    else:
        print("WARNING: No BERT scores found")
        bert_df = pd.DataFrame(columns=["email_id", "score_bert"])

    # Load Claude scores
    claude_path = Path("results/llm_scores_claude_v2.json")
    if claude_path.exists():
        claude_df = load_json_scores(claude_path, "score_claude")
        print(f"Claude scores: {len(claude_df)}")
    else:
        print("WARNING: No Claude scores found")
        claude_df = pd.DataFrame(columns=["email_id", "score_claude"])

    # Load GPT scores
    gpt_path = Path("results/llm_scores_gpt_v2.json")
    if gpt_path.exists():
        gpt_df = load_json_scores(gpt_path, "score_gpt")
        print(f"GPT scores: {len(gpt_df)}")
    else:
        print("WARNING: No GPT scores found")
        gpt_df = pd.DataFrame(columns=["email_id", "score_gpt"])

    # Merge
    scores = ground_truth.copy()
    if len(bert_df) > 0:
        scores = scores.merge(bert_df, on="email_id", how="left")
    else:
        scores["score_bert"] = float("nan")

    if len(claude_df) > 0:
        scores = scores.merge(claude_df, on="email_id", how="left")
    else:
        scores["score_claude"] = float("nan")

    if len(gpt_df) > 0:
        scores = scores.merge(gpt_df, on="email_id", how="left")
    else:
        scores["score_gpt"] = float("nan")

    # Report coverage
    print(f"\nCoverage:")
    for col in ["score_bert", "score_claude", "score_gpt"]:
        if col in scores.columns:
            n = scores[col].notna().sum()
            pct = n / len(scores) * 100
            print(f"  {col}: {n}/{len(scores)} ({pct:.1f}%)")

    # Report by split
    print(f"\nBy split:")
    for split in ["val", "cal", "test"]:
        s = scores[scores["split"] == split]
        coverage = {}
        for col in ["score_bert", "score_claude", "score_gpt"]:
            if col in s.columns:
                coverage[col] = f"{s[col].notna().sum()}/{len(s)}"
        print(f"  {split} ({len(s)}): {coverage}")

    # Save
    out_path = Path("scores.csv")
    scores.to_csv(out_path, index=False)
    print(f"\nSaved {len(scores)} rows to {out_path}")
    print(f"Columns: {list(scores.columns)}")


if __name__ == "__main__":
    main()
