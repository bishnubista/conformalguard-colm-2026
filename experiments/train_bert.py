#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "pyarrow>=14.0",
#   "transformers>=4.40",
#   "torch>=2.2",
#   "datasets>=2.18",
#   "accelerate>=0.28",
#   "scikit-learn>=1.3",
#   "numpy>=1.26",
# ]
# ///
"""
Train Scorer 1: fine-tune BERT-base-uncased on email subject + body text for
binary phishing classification (0=legit, 1=phishing).

Usage:
    uv run experiments/train_bert.py
    uv run experiments/train_bert.py --data-dir data/splits --model-dir models/bert_scorer
    uv run experiments/train_bert.py --epochs 5 --batch-size 8 --max-length 256

Outputs:
    models/bert_scorer/           -- final saved model + tokenizer
    models/bert_scorer/val_metrics.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Device detection ──────────────────────────────────────────────────────────

def detect_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        log.info(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        log.info("Device: Apple MPS")
    else:
        device = "cpu"
        log.info("Device: CPU (training will be slow)")
    return device


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Combine subject and body text with a separator token
    df["text"] = (
        df["subject"].fillna("").str.strip()
        + " [SEP] "
        + df["body_text"].fillna("").str.strip()
    )
    return df


def build_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int):
    """Tokenize a dataframe and return a HuggingFace Dataset."""
    from datasets import Dataset

    hf = Dataset.from_dict({
        "text": df["text"].tolist(),
        "label": df["label"].astype(int).tolist(),
    })

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    hf = hf.map(tokenize, batched=True, remove_columns=["text"])
    hf.set_format("torch")
    return hf


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Compute metrics for HuggingFace Trainer."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    logits, labels = eval_pred
    probs = _softmax(logits)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
    )

    device = detect_device()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load splits ──────────────────────────────────────────────────────────
    log.info("Loading data splits...")
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"

    if not train_path.exists():
        log.error(f"Train split not found: {train_path}")
        log.error("Run: uv run scripts/parse_and_split.py")
        sys.exit(1)

    train_df = load_split(train_path)
    val_df = load_split(val_path)

    log.info(f"Train: {len(train_df)} emails  ({(train_df['label']==0).sum()} legit, {(train_df['label']==1).sum()} phishing)")
    log.info(f"Val:   {len(val_df)} emails  ({(val_df['label']==0).sum()} legit, {(val_df['label']==1).sum()} phishing)")

    # Class weights for imbalanced data
    n_legit = (train_df["label"] == 0).sum()
    n_phish = (train_df["label"] == 1).sum()
    total = n_legit + n_phish
    # Weight = total / (n_classes * count_per_class)
    w_legit = total / (2 * n_legit)
    w_phish = total / (2 * n_phish)
    class_weights = torch.tensor([w_legit, w_phish], dtype=torch.float)
    log.info(f"Class weights: legit={w_legit:.3f}, phishing={w_phish:.3f}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    log.info("Tokenizing datasets...")
    train_ds = build_hf_dataset(train_df, tokenizer, args.max_length)
    val_ds = build_hf_dataset(val_df, tokenizer, args.max_length)
    log.info(f"  Train: {len(train_ds)} examples")
    log.info(f"  Val:   {len(val_ds)} examples")

    # ── Model ────────────────────────────────────────────────────────────────
    log.info(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "legit", 1: "phishing"},
        label2id={"legit": 0, "phishing": 1},
    )

    # ── Weighted loss trainer ─────────────────────────────────────────────────
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            import torch.nn as nn
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(
                weight=class_weights.to(logits.device)
            )
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # ── Training arguments ───────────────────────────────────────────────────
    # Determine fp16/bf16 availability
    use_fp16 = device == "cuda"
    use_bf16 = device == "mps"  # MPS prefers bf16

    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=50,
        report_to="none",
        seed=42,
        dataloader_num_workers=0,  # avoid multiprocessing issues on MPS/Windows
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Starting BERT fine-tuning (Scorer 1)")
    log.info(f"  Model:      {args.model_name}")
    log.info(f"  Epochs:     {args.epochs}")
    log.info(f"  Batch size: {args.batch_size}")
    log.info(f"  LR:         {args.lr}")
    log.info(f"  Max length: {args.max_length}")
    log.info(f"  Device:     {device}")
    log.info("=" * 60)

    train_result = trainer.train()

    log.info(f"Training complete: {train_result.metrics}")

    # ── Evaluate on val ──────────────────────────────────────────────────────
    log.info("Evaluating on validation set...")
    val_metrics = trainer.evaluate(val_ds)
    log.info("Validation metrics:")
    for k, v in val_metrics.items():
        if isinstance(v, float):
            log.info(f"  {k}: {v:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    log.info(f"Saving model to: {model_dir}")
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Save metadata alongside the model
    metadata = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "learning_rate": args.lr,
        "epochs_trained": int(train_result.metrics.get("epoch", args.epochs)),
        "train_samples": len(train_ds),
        "val_metrics": {k: round(v, 6) if isinstance(v, float) else v
                        for k, v in val_metrics.items()},
        "class_weights": {"legit": round(w_legit, 4), "phishing": round(w_phish, 4)},
        "input_format": "subject [SEP] body_text",
    }
    meta_path = model_dir / "training_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info(f"Training metadata saved: {meta_path}")

    # Print final summary
    log.info("")
    log.info("=" * 60)
    log.info("BERT Training Summary")
    log.info("=" * 60)
    log.info(f"  Model saved:        {model_dir}")
    log.info(f"  Val accuracy:       {val_metrics.get('eval_accuracy', 0):.4f}")
    log.info(f"  Val precision:      {val_metrics.get('eval_precision', 0):.4f}")
    log.info(f"  Val recall:         {val_metrics.get('eval_recall', 0):.4f}")
    log.info(f"  Val F1:             {val_metrics.get('eval_f1', 0):.4f}")
    log.info(f"  Val ROC-AUC:        {val_metrics.get('eval_roc_auc', 0):.4f}")
    log.info("")
    log.info("Next step:")
    log.info("  uv run experiments/calibrate_and_eval.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT for phishing detection (Scorer 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="data/splits",
        help="Directory containing train.parquet and val.parquet",
    )
    parser.add_argument(
        "--model-dir",
        default="models/bert_scorer",
        help="Output directory for the fine-tuned model",
    )
    parser.add_argument(
        "--model-name",
        default="bert-base-uncased",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token sequence length",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
