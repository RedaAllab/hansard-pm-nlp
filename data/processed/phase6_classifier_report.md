# Phase 6 - PM-attribution classifier

Generated: 2026-08-05T21:30:48.588674+00:00

Tests H1: 'a supervised classifier can attribute an anonymized speech excerpt to the correct PM at a rate significantly above chance' (CLAUDE.md). Document unit: one document per (PM, sitting date), same as Phase 5.

## Scope

3 classes: Boris Johnson, Rishi Sunak, Keir Starmer. Liz Truss excluded (5 documents total, 49-day tenure - too few for a train/test split or reliable per-class metrics; see split.py).

Train: 232 documents (earlier sittings per PM). Test: 59 documents (later sittings per PM, temporal split, test_size=0.2).

## Features

Lexical diversity (TTR, MTLD), readability (Flesch-Kincaid, mean words/sentence), hedging/certainty rates, individual function-word frequencies, and POS-tag distribution (spaCy) - see style_features.py. Class imbalance (Johnson 144 vs Sunak 63 vs Starmer 84 documents pre-split) handled via class_weight='balanced' in both models.

## Chance baselines

- Uniform random (1/3 classes): 0.333
- Always predict majority class (train set): 0.492

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic regression | 0.915 | 0.918 |
| HistGradientBoosting | 0.949 | 0.943 |

## Confusion matrix - logistic regression

| actual \ predicted | Boris Johnson | Keir Starmer | Rishi Sunak |
|---|---|---|---|
| Boris Johnson | 26 | 3 | 0 |
| Keir Starmer | 1 | 16 | 0 |
| Rishi Sunak | 0 | 1 | 12 |

## Confusion matrix - HistGradientBoosting

| actual \ predicted | Boris Johnson | Keir Starmer | Rishi Sunak |
|---|---|---|---|
| Boris Johnson | 28 | 0 | 1 |
| Keir Starmer | 1 | 16 | 0 |
| Rishi Sunak | 0 | 1 | 12 |

## Feature importance (permutation, top 15)

Model-agnostic (permutation_importance on the test set), so both models are ranked the same way rather than comparing a linear model's coefficients to a tree ensemble's split gains.

### Logistic regression

- fw_had: 0.0221
- fw_been: 0.0206
- fw_was: 0.0199
- fw_not: 0.0177
- pos_SCONJ: 0.0112
- fw_of: 0.0080
- fw_those: 0.0074
- fw_a: 0.0066
- fw_but: 0.0057
- fw_were: 0.0051
- pos_INTJ: 0.0044
- fw_they: 0.0044
- fw_to: 0.0044
- fw_i: 0.0033
- pos_ADJ: 0.0033

### HistGradientBoosting

- mean_words_per_sentence: 0.0536
- pos_INTJ: 0.0488
- hedge_rate: 0.0485
- fw_so: 0.0326
- fw_of: 0.0286
- fw_what: 0.0238
- pos_PROPN: 0.0230
- fw_because: 0.0158
- fw_must: 0.0131
- fw_to: 0.0131
- mtld: 0.0130
- fw_a: 0.0128
- pos_PART: 0.0105
- pos_NOUN: 0.0105
- fw_i: 0.0104
