# ConformalGuard Camera-Ready Artifact

This artifact accompanies the COLM 2026 camera-ready manuscript
"ConformalGuard: False-Positive-Controlled Action Gating for LM Classifiers."
It is a privacy-preserving verification and reconstruction artifact, not a
bit-for-bit recipe for rerunning mutable model-provider APIs.

[Paper on arXiv](https://arxiv.org/abs/2610.14365)

## What changed after the accepted submission

The accepted v7 supplementary archive is preserved unchanged at
`supplementary_code.zip` in the paper repository. This camera-ready export is
built as a reviewed overlay on that immutable snapshot. It fixes two stale
derived files and adds the post-submission stochastic-averaging result:

- `data/splits_v2/split_summary.json` now describes the actual submitted
  Parquet splits: 39,935 rows, not the 39,951-row pre-leakage-cleanup state.
- `results/table2_low_prevalence_camera_ready.csv` reports the manuscript's
  full-test recall and per-10,000 workload quantities, including paired
  bootstrap intervals.
- `results/full_stochastic_averaging_summary.json` contains the full-test
  GPT-4o one-run versus three-run averaging result.

See `PROVENANCE.md` for hashes, lineage, and the exact reconciliation.

## Privacy-preserving export

The public camera-ready export intentionally omits the Parquet files and raw
email bodies. Several upstream corpora contain real messages and retain their
own copyright or reuse terms. The export instead includes:

- pinned download and parsing scripts;
- a deterministic cross-split leakage cleanup;
- the corrected split manifest and source-file SHA-256 hashes;
- pseudonymous, source-prefixed score records with labels and split IDs;
- all reported result tables and verification scripts.

See `THIRD_PARTY_DATA.md` before downloading or processing the corpora.
See `PROVIDER_OUTPUTS.md` for the scope and reuse status of the numeric
provider-score files.

## Quick verification

From the exported artifact root:

```bash
python experiments/assemble_scores.py
python experiments/generate_camera_ready_robustness.py
python experiments/generate_camera_ready_low_prevalence.py
python scripts/verify_camera_ready_artifact.py
```

These commands use the Python standard library. The robustness generator
reproduces the calibration sensitivity, source-shift, within-source, and
randomized tie-breaking analyses from the bundled pseudonymous scores.

The accepted-submission calibration runner is retained as
`experiments/run_calibration_comparison_submission.py` for provenance and for
the main comparison. Its original subsampled low-prevalence routine is
superseded by `generate_camera_ready_low_prevalence.py`; do not use the
submission runner to regenerate the camera-ready workload table.

## Rebuild the dataset locally

Raw data is not required to verify the reported tables. To reconstruct the
preprocessed splits:

```bash
bash scripts/download_sources.sh data/raw
uv run scripts/parse_and_split.py \
  --raw-dir data/raw \
  --out-dir data/splits_precleanup \
  --cal-size 2000 \
  --max-phishing 100000 \
  --max-legit 100000
uv run scripts/remove_cross_split_duplicates.py \
  --input-dir data/splits_precleanup \
  --output-dir data/splits_v2
```

The final files should match the counts and SHA-256 hashes in
`data/splits_v2/`. Upstream hosting can change; if a download hash fails, do
not silently substitute a newer snapshot.

## Maintainer-only release build

The following command is for maintainers with the private paper repository;
it is not required for public verification:

```bash
bash camera-ready/v10/artifact/scripts/build_camera_ready_artifact.sh
```

The script creates `camera-ready/v10/conformalguard-camera-ready-artifact.zip`
without modifying the accepted submission snapshot.
