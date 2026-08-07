# ConformalGuard Data Quality Audit Report

**Audit date:** 2026-03-25
**Camera-ready reconciliation:** 2026-07-22
**Dataset:** EPVME-excluded `splits_v2`

## Final split counts

| Split | Rows | Legitimate | Phishing/Spam | Positive rate |
|---|---:|---:|---:|---:|
| Train | 26,202 | 11,937 | 14,265 | 54.4% |
| Validation | 3,742 | 1,705 | 2,037 | 54.4% |
| Calibration | 2,000 | 2,000 | 0 | 0.0% |
| Test | 7,991 | 3,911 | 4,080 | 51.1% |
| **Total** | **39,935** | **19,553** | **20,382** | **51.0%** |

## Checks

- EPVME rows: 0.
- Empty bodies: 0.
- Bodies shorter than 50 characters: 0.
- Cross-split normalized-body duplicates after cleanup: 0.
- Test and calibration rows changed by leakage cleanup: 0.
- Source-confusion classifier: 97.7% +/- 0.1% accuracy with TF-IDF and
  logistic regression under five-fold cross-validation.

Before leakage cleanup, the split files contained 39,951 rows and 39,918
unique normalized body hashes. Fifteen normalized hashes crossed split
boundaries. Keeping the highest-priority split (`test`, `cal`, `val`, `train`)
removed 16 lower-priority rows: 13 from train and 3 from validation.

## Positive-class composition

| Category | Sources | Count | Share of positives |
|---|---|---:|---:|
| True phishing | Nazario, Phishing Pot | 5,844 | 28.7% |
| Spam-as-positive | Enron spam, SpamAssassin spam | 14,538 | 71.3% |

This composition makes the empirical task malicious/unwanted-email detection,
not pure phishing detection. Source-specific vocabulary is a known limitation;
all calibration methods are compared on the same frozen score distributions.
