#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OVERLAY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAPER_ROOT="$(cd "$OVERLAY_ROOT/../../.." && pwd)"
BASE="$PAPER_ROOT/anonymous-repo"
OUTPUT="${1:-$OVERLAY_ROOT/../conformalguard-camera-ready-artifact.zip}"
STAGING_DIR="$(mktemp -d)"
PACKAGE="$STAGING_DIR/conformalguard-camera-ready-artifact"

cleanup() {
  find "$STAGING_DIR" -depth -delete
}
trap cleanup EXIT

mkdir -p "$PACKAGE"
rsync -a \
  --exclude='README.md' \
  --exclude='DATA_QUALITY_AUDIT_REPORT.md' \
  --exclude='data/' \
  --exclude='results/table2_low_prevalence.csv' \
  --exclude='experiments/run_calibration_comparison.py' \
  --exclude='scripts/download_sources.sh' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "$BASE/" "$PACKAGE/"
cp "$BASE/experiments/run_calibration_comparison.py" \
  "$PACKAGE/experiments/run_calibration_comparison_submission.py"
rsync -a --exclude='__pycache__/' --exclude='*.pyc' "$OVERLAY_ROOT/" "$PACKAGE/"

STOCHASTIC="$PAPER_ROOT/reviews/openreview-2026-05-response/full-stochastic-averaging"
cp "$STOCHASTIC/full_stochastic_averaging_summary.json" \
  "$PACKAGE/results/full_stochastic_averaging_summary.json"

python3 "$PACKAGE/experiments/generate_camera_ready_low_prevalence.py"
python3 "$PACKAGE/scripts/verify_camera_ready_artifact.py"

(
  cd "$PACKAGE"
  find . -type f ! -name SHA256SUMS.txt -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 > SHA256SUMS.txt
  find . -exec touch -t 202607220000 {} +
)

ARCHIVE_TMP="$STAGING_DIR/conformalguard-camera-ready-artifact.zip"
(
  cd "$STAGING_DIR"
  find conformalguard-camera-ready-artifact -print \
    | LC_ALL=C sort \
    | zip -X -q "$ARCHIVE_TMP" -@
)
mkdir -p "$(dirname "$OUTPUT")"
mv -f "$ARCHIVE_TMP" "$OUTPUT"
shasum -a 256 "$OUTPUT"
