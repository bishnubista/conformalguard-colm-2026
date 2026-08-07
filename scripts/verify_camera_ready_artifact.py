#!/usr/bin/env python3
"""Verify camera-ready artifact counts, hashes, and reported outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


manifest = json.loads((ROOT / "data/splits_v2/split_summary.json").read_text())
require(manifest["total"] == 39935, "split total must be 39,935")
require(manifest["legitimate"] == 19553, "legitimate total mismatch")
require(manifest["positive"] == 20382, "positive total mismatch")
require(sum(item["total"] for item in manifest["splits"].values()) == 39935, "split sum")
require(sum(manifest["sources"].values()) == 39935, "source sum")
require(manifest["sources"]["phishing_pot"] == 4393, "Phishing Pot final count")
require(manifest["leakage_cleanup"]["removed_total"] == 16, "leakage cleanup count")
require(not list(ROOT.rglob("*.parquet")), "public export must not contain Parquet email data")

scores_path = ROOT / "results/bert_scores_v2.json"
require(
    hashlib.sha256(scores_path.read_bytes()).hexdigest()
    == "76a85361ee46bf2a321b5121030997a853f90b8acb9d7ed935b05cde2885ff75",
    "BERT score file hash",
)
scores = json.loads(scores_path.read_text())
counts = {
    split: sum(row["split"] == split for row in scores)
    for split in ("val", "cal", "test")
}
require(counts == {"val": 3742, "cal": 2000, "test": 7991}, "score split counts")

with (ROOT / "results/table2_low_prevalence_camera_ready.csv").open(newline="") as handle:
    low_prevalence = {row["method"]: row for row in csv.DictReader(handle)}
require(round(float(low_prevalence["Raw threshold"]["fp_per_10k"])) == 83, "raw FP/10K")
require(round(float(low_prevalence["Conformal"]["fp_per_10k"])) == 58, "conformal FP/10K")
require(round(float(low_prevalence["Raw threshold"]["fp_ci_low"])) == 55, "raw CI low")
require(round(float(low_prevalence["Raw threshold"]["fp_ci_high"])) == 113, "raw CI high")
require(round(float(low_prevalence["Conformal"]["fp_ci_low"])) == 35, "conformal CI low")
require(round(float(low_prevalence["Conformal"]["fp_ci_high"])) == 83, "conformal CI high")

stochastic = json.loads(
    (ROOT / "results/full_stochastic_averaging_summary.json").read_text()
)
require(len(stochastic) == 2, "stochastic summary row count")
require(stochastic[1]["complete_by_split"] == {"cal": 2000, "val": 3742, "test": 7991}, "stochastic coverage")
require(abs(stochastic[1]["conformal"]["recall"] - 0.7318627450980392) < 1e-12, "avg3 recall")
require(abs(stochastic[1]["conformal"]["fpr"] - 0.005113781641523907) < 1e-12, "avg3 FPR")

print("PASS: camera-ready artifact is internally consistent")
