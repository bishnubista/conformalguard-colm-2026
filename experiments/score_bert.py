#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "transformers>=4.40",
#   "torch>=2.2",
#   "numpy>=1.26",
#   "tqdm>=4.65",
# ]
# ///
"""
Score emails with a fine-tuned BERT model. Outputs P(phishing) for each email.

Usage:
    uv run experiments/score_bert.py --model-dir models/bert_scorer_v2 --data-dir data/splits_v2
    uv run experiments/score_bert.py --splits val cal test

Output: results/bert_scores_v2.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def score_emails(model, tokenizer, texts, device, batch_size=64, max_length=512):
    """Score a list of texts, return P(phishing) for each."""
    model.eval()
    all_probs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1]  # P(phishing)
            all_probs.extend(probs.cpu().numpy().tolist())

    return all_probs


def main():
    parser = argparse.ArgumentParser(description="Score emails with BERT")
    parser.add_argument("--model-dir", default="models/bert_scorer_v2")
    parser.add_argument("--data-dir", default="data/splits_v2")
    parser.add_argument("--output", default="results/bert_scores_v2.json")
    parser.add_argument("--splits", nargs="+", default=["val", "cal", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)

    # Load metadata
    meta_path = model_dir / "training_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        max_length = meta.get("max_length", 512)
        print(f"Model: {meta.get('model_name')}, max_length={max_length}")
    else:
        max_length = 512

    # Score each split
    all_results = []
    for split in args.splits:
        print(f"\nScoring {split} split...")
        df = pd.read_parquet(data_dir / f"{split}.parquet")
        texts = (
            df["subject"].fillna("").str.strip()
            + " [SEP] "
            + df["body_text"].fillna("").str.strip()
        ).tolist()

        scores = score_emails(model, tokenizer, texts, device, args.batch_size, max_length)

        for idx, (_, row) in enumerate(df.iterrows()):
            all_results.append({
                "email_id": row["email_id"],
                "split": split,
                "label": int(row["label"]),
                "source": row.get("source", ""),
                "score_bert": round(scores[idx], 6),
            })

        # Summary
        scores_arr = np.array(scores)
        labels = df["label"].values
        preds = (scores_arr >= 0.5).astype(int)
        acc = (preds == labels).mean()
        print(f"  {split}: {len(df)} emails, accuracy={acc:.4f}, "
              f"mean_score={scores_arr.mean():.4f}, "
              f"pos_mean={scores_arr[labels==1].mean():.4f}, "
              f"neg_mean={scores_arr[labels==0].mean():.4f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} scores to {output_path}")


if __name__ == "__main__":
    main()
