# Phase 4 - Sentiment, hedging, certainty

Generated: 2026-08-05T19:42:41.908969+00:00

Descriptive only. `vader_compound` in [-1, 1] (lexicon-based); `transformer_score` in [-1, 1], signed by the distilbert-base-uncased-finetuned-sst-2-english label (contextual, binary). `sentiment_sign_agreement` is the share of contributions where both methods agree on positive vs. negative.

## Per PM

| PM | N | Mean VADER | Mean transformer | Sign agreement | Mean hedge rate | Mean booster rate | Mean net certainty |
|---|---|---|---|---|---|---|---|
| Boris Johnson | 5459 | 0.561 | 0.534 | 76.4% | 0.0113 | 0.0164 | 0.0051 |
| Keir Starmer | 2896 | 0.465 | 0.403 | 74.3% | 0.0073 | 0.0158 | 0.0085 |
| Liz Truss | 123 | 0.617 | 0.539 | 74.0% | 0.0046 | 0.0206 | 0.0160 |
| Rishi Sunak | 2195 | 0.552 | 0.622 | 78.9% | 0.0060 | 0.0137 | 0.0077 |

## Hedging by debate type

| Is PMQs | N | Mean hedge rate | Mean net certainty |
|---|---|---|---|
| False | 4597 | 0.0102 | 0.0073 |
| True | 6076 | 0.0082 | 0.0062 |
