# Camera-Ready Analysis Tables

Generated from the bundled score artifacts.

## Dataset Snapshot

- Rows in `scores.csv`: 13733
- Test positives / legitimate: 4080 / 3911
- Calibration legitimate examples: 2000
- Top sources: enron_enron2_ham=1650, spamassassin_ham=1557, enron_enron3_ham=1416, enron_enron1_ham=1336, phishing_pot=1255, enron_enron4_spam=1076

## Table A — Main Calibration Comparison

| LM | Calibration | Recall | FPR | Precision | F1 | FPR viol. |
|---|---|---|---|---|---|---|
| BERT (fine-tuned) | Raw threshold | 0.9963 | 0.0084 | 0.9919 | 0.9941 | -0.0016 |
| BERT (fine-tuned) | Temp. scaling | 0.9963 | 0.0084 | 0.9919 | 0.9941 | -0.0016 |
| BERT (fine-tuned) | Isotonic reg. | 0.9973 | 0.0125 | 0.9881 | 0.9927 | 0.0025 |
| BERT (fine-tuned) | Conformal | 0.9953 | 0.0059 | 0.9944 | 0.9949 | -0.0041 |
| Claude Sonnet 4 | Raw threshold | 0.9377 | 0.0399 | 0.9608 | 0.9491 | 0.0299 |
| Claude Sonnet 4 | Temp. scaling | 0.9377 | 0.0399 | 0.9608 | 0.9491 | 0.0299 |
| Claude Sonnet 4 | Isotonic reg. | 0.9377 | 0.0399 | 0.9608 | 0.9491 | 0.0299 |
| Claude Sonnet 4 | Conformal | 0.8755 | 0.0079 | 0.9914 | 0.9298 | -0.0021 |
| GPT-4o | Raw threshold | 0.887 | 0.0153 | 0.9837 | 0.9329 | 0.0053 |
| GPT-4o | Temp. scaling | 0.887 | 0.0153 | 0.9837 | 0.9329 | 0.0053 |
| GPT-4o | Isotonic reg. | 0.887 | 0.0153 | 0.9837 | 0.9329 | 0.0053 |
| GPT-4o | Conformal | 0.564 | 0.0008 | 0.9987 | 0.7209 | -0.0092 |

## Table B — Conformal Threshold Sensitivity From Current Scores

| LM | alpha | Recall | FPR | Precision | F1 |
|---|---|---|---|---|---|
| BERT | 0.001 | 0.9681 | 0.0010 | 0.9990 | 0.9833 |
| BERT | 0.005 | 0.9924 | 0.0033 | 0.9968 | 0.9946 |
| BERT | 0.01 | 0.9953 | 0.0059 | 0.9944 | 0.9949 |
| BERT | 0.05 | 0.9993 | 0.0519 | 0.9526 | 0.9754 |
| BERT | 0.1 | 0.9998 | 0.0969 | 0.9150 | 0.9555 |
| Claude Sonnet 4 | 0.001 | 0.0395 | 0.0000 | 1.0000 | 0.0759 |
| Claude Sonnet 4 | 0.005 | 0.7294 | 0.0018 | 0.9977 | 0.8427 |
| Claude Sonnet 4 | 0.01 | 0.8755 | 0.0079 | 0.9914 | 0.9298 |
| Claude Sonnet 4 | 0.05 | 0.8755 | 0.0079 | 0.9914 | 0.9298 |
| Claude Sonnet 4 | 0.1 | 0.9701 | 0.0603 | 0.9437 | 0.9567 |
| GPT-4o | 0.001 | 0.5640 | 0.0008 | 0.9987 | 0.7209 |
| GPT-4o | 0.005 | 0.5640 | 0.0008 | 0.9987 | 0.7209 |
| GPT-4o | 0.01 | 0.5640 | 0.0008 | 0.9987 | 0.7209 |
| GPT-4o | 0.05 | 0.8870 | 0.0153 | 0.9837 | 0.9329 |
| GPT-4o | 0.1 | 0.9534 | 0.0672 | 0.9367 | 0.9450 |

## Table C — Oracle Matched-FPR Sanity Check

Raw thresholding can match conformal decisions if it is allowed to choose the threshold using test labels. This table should be used to clarify that conformal's advantage is pre-deployment threshold validity, not dominance over an oracle threshold.

| LM | Conformal recall | Conformal FPR | Oracle raw threshold | Oracle raw recall | Oracle raw FPR | Oracle raw F1 |
|---|---|---|---|---|---|---|
| BERT | 0.9953 | 0.0059 | 0.005657 | 0.9953 | 0.0059 | 0.9949 |
| Claude Sonnet 4 | 0.8755 | 0.0079 | 0.85 | 0.8755 | 0.0079 | 0.9298 |
| GPT-4o | 0.5640 | 0.0008 | 0.95 | 0.5640 | 0.0008 | 0.7209 |

## Table D — Balanced-Validation Calibration Ablation

Temperature scaling and isotonic regression are fit on a balanced validation subset, then thresholded for alpha=0.01 using the legitimate half of that subset. This answers whether adding positives to the calibration fit fixes the LLM-score FPR issue.

| LM | Method | Recall | FPR | Precision | F1 | Fit detail |
|---|---|---|---|---|---|---|
| BERT | Temp. scaling balanced-val | 0.9963 | 0.0084 | 0.9919 | 0.9941 | T=1.574 |
| BERT | Isotonic balanced-val | 0.9973 | 0.0130 | 0.9876 | 0.9924 | threshold=0.182 |
| Claude Sonnet 4 | Temp. scaling balanced-val | 0.9377 | 0.0399 | 0.9608 | 0.9491 | T=0.626 |
| Claude Sonnet 4 | Isotonic balanced-val | 0.9377 | 0.0399 | 0.9608 | 0.9491 | threshold=0.904 |
| GPT-4o | Temp. scaling balanced-val | 0.8870 | 0.0153 | 0.9837 | 0.9329 | T=0.661 |
| GPT-4o | Isotonic balanced-val | 0.8870 | 0.0153 | 0.9837 | 0.9329 | threshold=0.993 |

## Table F — Legitimate-Source Shift Stress Test

Calibration uses legitimate examples from one source family and evaluates FPR on legitimate examples from another. Positive-class recall is not measured here; this isolates exchangeability/FPR behavior on legitimate traffic.

| LM | Calibration source | n_cal | Test legitimate source | n_test | Observed FPR | FPR - alpha |
|---|---|---|---|---|---|---|
| BERT | enron_ham | 2949 | enron_ham | 3110 | 0.0064 | -0.0036 |
| BERT | enron_ham | 2949 | spamassassin_ham | 801 | 0.0212 | 0.0112 |
| BERT | spamassassin_ham | 756 | enron_ham | 3110 | 0.0032 | -0.0068 |
| BERT | spamassassin_ham | 756 | spamassassin_ham | 801 | 0.0112 | 0.0012 |
| Claude Sonnet 4 | enron_ham | 2949 | enron_ham | 3110 | 0.0084 | -0.0016 |
| Claude Sonnet 4 | enron_ham | 2949 | spamassassin_ham | 801 | 0.0062 | -0.0038 |
| Claude Sonnet 4 | spamassassin_ham | 756 | enron_ham | 3110 | 0.0084 | -0.0016 |
| Claude Sonnet 4 | spamassassin_ham | 756 | spamassassin_ham | 801 | 0.0062 | -0.0038 |
| GPT-4o | enron_ham | 2949 | enron_ham | 3110 | 0.0010 | -0.0090 |
| GPT-4o | enron_ham | 2949 | spamassassin_ham | 801 | 0.0000 | -0.0100 |
| GPT-4o | spamassassin_ham | 756 | enron_ham | 3110 | 0.0174 | 0.0074 |
| GPT-4o | spamassassin_ham | 756 | spamassassin_ham | 801 | 0.0075 | -0.0025 |

## Table G — Low-Prevalence Workload With Correct Per-10K Scaling

The earlier `fp_mean` field in `table2_low_prevalence.csv` is the raw false-positive count over 3,911 legitimate test emails. This table rescales FPR to a 10,000-email inbox at 2% phishing prevalence, i.e. 9,800 legitimate emails. Brackets are paired bootstrap 95% intervals over legitimate test emails. The raw-minus-conformal reduction is 25 FP per 10K emails, 95% CI [10, 43].

| Calibration | Recall | FPR | Precision at 2% | FP per 10K emails [95% CI] |
|---|---|---|---|---|
| Raw threshold | 0.9963 | 0.0084 | 0.7067 | 83 [55, 113] |
| Temp. scaling | 0.9963 | 0.0084 | 0.7067 | 83 [55, 113] |
| Conformal | 0.9953 | 0.0059 | 0.7755 | 58 [35, 83] |

## Table H — Randomized Conformal Tie-Breaking Baseline

Randomized conformal tie-breaking can split large tied score blocks, but this makes action assignment stochastic. `Decision depends on U` is the fraction of test examples whose p-value interval straddles alpha; `seed disagreement` is the average fraction of test examples whose decision differs from seed 0 across 99 other random seeds. This table directly answers whether randomized tie-breaking is a free fix for coarse LLM scores.

| LM | alpha | Recall | FPR | Precision | F1 | Decision depends on U | Seed disagreement |
|---|---|---|---|---|---|---|---|
| BERT | 0.001 | 0.968 ± 0.000 | 0.001 ± 0.000 | 0.999 | 0.983 | 0.1% | 0.0% |
| BERT | 0.005 | 0.992 ± 0.000 | 0.003 ± 0.000 | 0.997 | 0.995 | 0.0% | 0.0% |
| BERT | 0.01 | 0.995 ± 0.000 | 0.006 ± 0.000 | 0.994 | 0.995 | 0.0% | 0.0% |
| BERT | 0.05 | 0.999 ± 0.000 | 0.052 ± 0.000 | 0.952 | 0.975 | 0.1% | 0.0% |
| BERT | 0.1 | 1.000 ± 0.000 | 0.098 ± 0.000 | 0.914 | 0.955 | 0.1% | 0.0% |
| Claude Sonnet 4 | 0.001 | 0.308 ± 0.006 | 0.000 ± 0.000 | 0.999 | 0.471 | 34.3% | 16.4% |
| Claude Sonnet 4 | 0.005 | 0.817 ± 0.003 | 0.006 ± 0.001 | 0.994 | 0.896 | 7.8% | 3.8% |
| Claude Sonnet 4 | 0.01 | 0.881 ± 0.001 | 0.010 ± 0.001 | 0.989 | 0.932 | 4.7% | 0.7% |
| Claude Sonnet 4 | 0.05 | 0.937 ± 0.000 | 0.040 ± 0.000 | 0.961 | 0.949 | 6.2% | 0.1% |
| Claude Sonnet 4 | 0.1 | 0.977 ± 0.001 | 0.089 ± 0.003 | 0.920 | 0.947 | 5.6% | 2.4% |
| GPT-4o | 0.001 | 0.576 ± 0.002 | 0.001 ± 0.000 | 0.998 | 0.730 | 17.2% | 1.3% |
| GPT-4o | 0.005 | 0.675 ± 0.004 | 0.006 ± 0.001 | 0.992 | 0.803 | 17.2% | 7.6% |
| GPT-4o | 0.01 | 0.800 ± 0.004 | 0.011 ± 0.001 | 0.986 | 0.883 | 17.2% | 7.0% |
| GPT-4o | 0.05 | 0.927 ± 0.002 | 0.047 ± 0.002 | 0.953 | 0.940 | 5.9% | 2.8% |
| GPT-4o | 0.1 | 0.971 ± 0.001 | 0.095 ± 0.001 | 0.914 | 0.942 | 2.6% | 0.7% |

