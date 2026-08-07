#!/usr/bin/env python3
"""Assemble bundled scorer outputs without requiring the raw email corpus."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "scores.csv"


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def score_map(path: Path) -> dict[str, float]:
    rows = load(path)
    return {
        row["email_id"]: float(row.get("score", row.get("score_bert", 0.5)))
        for row in rows
    }


def main() -> None:
    bert_rows = load(RESULTS / "bert_scores_v2.json")
    claude = score_map(RESULTS / "llm_scores_claude_v2.json")
    gpt = score_map(RESULTS / "llm_scores_gpt_v2.json")

    bert_ids = {row["email_id"] for row in bert_rows}
    if set(claude) != bert_ids or set(gpt) != bert_ids:
        raise SystemExit("Score files do not cover the same pseudonymous IDs")

    fields = [
        "email_id",
        "split",
        "label",
        "source",
        "score_bert",
        "score_claude",
        "score_gpt",
    ]
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in bert_rows:
            email_id = row["email_id"]
            writer.writerow(
                {
                    "email_id": email_id,
                    "split": row["split"],
                    "label": row["label"],
                    "source": row["source"],
                    "score_bert": row["score_bert"],
                    "score_claude": claude[email_id],
                    "score_gpt": gpt[email_id],
                }
            )

    print(f"Wrote {len(bert_rows)} joined score rows to {OUT}")


if __name__ == "__main__":
    main()
