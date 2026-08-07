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

provider_score_hashes = {
    "llm_scores_claude_v2.json": "33cc08d8d828b874b3b52ba8215cd72b8a6a16ac26271279d28e87fa1dda720b",
    "llm_scores_gpt_v2.json": "2a945a19fab6e9f1a164f45de9a5be2a4fa50375013752be0f3e4ef963ed104b",
}
for filename, expected_hash in provider_score_hashes.items():
    path = ROOT / "results" / filename
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, f"{filename} hash")

analysis = (ROOT / "results/camera_ready_analysis_tables.md").read_text()
expected_analysis_rows = (
    "| BERT | Temp. scaling balanced-val | 0.9963 | 0.0084 | 0.9919 | 0.9941 | T=1.574 |",
    "| Claude Sonnet 4 | Temp. scaling balanced-val | 0.9377 | 0.0399 | 0.9608 | 0.9491 | T=0.626 |",
    "| GPT-4o | 0.01 | 0.5640 | 0.0008 | 0.9987 | 0.7209 |",
    "| GPT-4o | spamassassin_ham | 756 | enron_ham | 3110 | 0.0174 | 0.0074 |",
    "| GPT-4o | 0.01 | 0.800 ± 0.004 | 0.011 ± 0.001 | 0.986 | 0.883 | 17.2% | 7.0% |",
)
for row in expected_analysis_rows:
    require(row in analysis, f"camera-ready analysis row: {row}")

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
