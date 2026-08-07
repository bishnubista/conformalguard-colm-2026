# Camera-Ready Artifact Provenance

## Immutable accepted-submission baseline

- Accepted manuscript baseline: v7.
- Submitted supplementary archive: `supplementary_code.zip`.
- Submitted archive SHA-256:
  `f9a1c0855ddf44d16610a30ea53583d645f502c425e5fe048b34eeaa06a0d9a0`.
- Archive root: `anonymous-repo/`.
- Archived mirror commit: `a3369382` in the private
  `conformalguard-supplementary` repository.

The archive and `anonymous-repo/` directory remain unchanged. Camera-ready
corrections are applied only by this overlay/export.

## Split-manifest reconciliation

The submitted Parquet files contain 39,935 rows:

| Split | Total | Legitimate | Positive |
|---|---:|---:|---:|
| Train | 26,202 | 11,937 | 14,265 |
| Validation | 3,742 | 1,705 | 2,037 |
| Calibration | 2,000 | 2,000 | 0 |
| Test | 7,991 | 3,911 | 4,080 |

The submitted `split_summary.json` was generated before a documented
cross-split leakage cleanup and still reported 39,951 rows. The cleanup hashes
`lower(strip(body_text))` with SHA-256, retains all rows in the highest-priority
split (`test`, then `cal`, then `val`, then `train`), and removes duplicates
only from lower-priority splits. It removed 13 train rows and 3 validation
rows. Test and calibration were unchanged.

The corrected manifest also fixes the affected per-source counts, including
Phishing Pot from 4,395 to 4,393 final rows. The exact removed pseudonymous
email IDs are recorded in
`data/splits_v2/removed_cross_split_email_ids.txt`.

## Low-prevalence table reconciliation

The accepted artifact's `table2_low_prevalence.csv` reported raw false-positive
counts over the 3,911 legitimate test emails (23 conformal and 33 raw) while
its filename/context was later read as a per-10,000 workload result.

The camera-ready generator:

1. selects the raw threshold at the 0.99 linear quantile of validation
   legitimate BERT scores;
2. computes deterministic conformal p-values from all 2,000 legitimate
   calibration scores;
3. measures recall on all 4,080 positive test examples;
4. scales FPR to 9,800 legitimate emails in a 10,000-email inbox at 2%
   prevalence; and
5. uses 5,000 paired bootstrap resamples of the legitimate test examples with
   seed `20260527`.

This yields 83 [55, 113] raw versus 58 [35, 83] conformal false positives per
10,000 emails and a reduction of 25 [10, 43].

## Post-submission stochastic averaging

The exported `full_stochastic_averaging_summary.json` is copied from the
paper-local OpenReview response evidence. It covers all 2,000 calibration,
3,742 validation, and 7,991 test examples. Three-run GPT-4o averaging raises
deterministic conformal recall from 0.2848 to 0.7319 at FPR 0.0051.

## Source lineage

- Active private experiment source:
  `bishnubista/conformalguard-experiments`.
- Verified branch: `feat/calibration-comparison-v2`.
- Verified commit:
  `834be74bba122ff202297769302979891930d1a5`.
- Phishing Pot source pin:
  `014392a2dfcdfae50a12841e9258b481888275a9`.
