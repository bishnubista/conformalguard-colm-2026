#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.0",
#   "pyarrow>=14.0",
# ]
# ///
"""Remove normalized-body leakage across splits without altering high-priority splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SPLITS = ("train", "val", "cal", "test")
PRIORITY = {"test": 0, "cal": 1, "val": 2, "train": 3}


def normalized_hash(body: object) -> str:
    text = "" if body is None else str(body)
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise SystemExit("Use a distinct output directory; input files are never overwritten.")

    frames = []
    for split in SPLITS:
        frame = pd.read_parquet(args.input_dir / f"{split}.parquet")
        frame["_split"] = split
        frame["_normalized_body_hash"] = frame["body_text"].map(normalized_hash)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)

    remove_indices: set[int] = set()
    for _, group in combined.groupby("_normalized_body_hash", sort=False):
        present = set(group["_split"])
        if len(present) < 2:
            continue
        keep_split = min(present, key=PRIORITY.__getitem__)
        remove_indices.update(group[group["_split"] != keep_split].index)

    cleaned = combined.drop(index=sorted(remove_indices))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "total": int(len(cleaned)),
        "legitimate": int((cleaned["label"] == 0).sum()),
        "positive": int((cleaned["label"] == 1).sum()),
        "splits": {},
        "sources": {
            key: int(value)
            for key, value in cleaned["source"].value_counts().sort_index().items()
        },
        "leakage_cleanup": {
            "normalization": "lower(strip(body_text))",
            "hash": "sha256",
            "split_priority": ["test", "cal", "val", "train"],
            "removed_total": len(remove_indices),
        },
    }
    for split in SPLITS:
        frame = cleaned[cleaned["_split"] == split].drop(
            columns=["_split", "_normalized_body_hash"]
        )
        frame.to_parquet(args.output_dir / f"{split}.parquet", index=False)
        summary["splits"][split] = {
            "total": int(len(frame)),
            "legitimate": int((frame["label"] == 0).sum()),
            "positive": int((frame["label"] == 1).sum()),
        }
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Removed {len(remove_indices)} lower-priority cross-split rows")
    print(f"Wrote {len(cleaned)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
