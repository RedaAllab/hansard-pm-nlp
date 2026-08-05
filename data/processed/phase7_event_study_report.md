# Phase 7 - Event-study regressions

Generated: 2026-08-05T22:19:16.166118+00:00

Tests H2 (crisis affect) and H3 (party interaction) per CLAUDE.md §2. Unit: one row per (PM, sitting date), 296 sittings after the 50-word floor (event_study.py) - unlike Phase 6's classifier, Liz Truss is included here, since her tenure is the only one overlapping the mini-budget crisis window.

## H2 - crisis dummies (PM fixed effects, HC3 robust SE)

`DV ~ PM fixed effects + one dummy per named crisis`. Each coefficient is the shift in that PM's own score during the named crisis, relative to their own baseline. Benjamini-Hochberg FDR correction applied across all 12 tests (4 crises x 3 dependent variables) together.

Sittings per crisis window: covid19=67, mini_budget=1, ukraine_invasion=11, labour_leadership_crisis=9. **The mini-budget window has exactly 1 sitting** (Liz Truss, 2022-10-12) - its coefficient's standard error (~3.0 for VADER, a score bounded in [-1, 1]) reflects that single-observation identification, not a genuine null effect. Read the mini-budget row as uninterpretable given available data, not as evidence against H2 for that crisis specifically.

| dv | crisis | coef | se | pvalue | pvalue_bh | significant_bh |
|---|---|---|---|---|---|---|
| vader_compound | covid19 | 0.0410 | 0.0235 | 0.0808 | 0.4847 | False |
| vader_compound | mini_budget | -0.0191 | 3.0023 | 0.9949 | 0.9949 | False |
| vader_compound | ukraine_invasion | 0.0047 | 0.0576 | 0.9352 | 0.9949 | False |
| vader_compound | labour_leadership_crisis | 0.0124 | 0.0435 | 0.7756 | 0.9949 | False |
| transformer_score | covid19 | 0.0757 | 0.0414 | 0.0675 | 0.4847 | False |
| transformer_score | mini_budget | -0.3132 | 1.7542 | 0.8583 | 0.9949 | False |
| transformer_score | ukraine_invasion | 0.0521 | 0.0784 | 0.5061 | 0.9949 | False |
| transformer_score | labour_leadership_crisis | 0.0741 | 0.0878 | 0.3985 | 0.9949 | False |
| net_certainty | covid19 | -0.0016 | 0.0012 | 0.1988 | 0.7950 | False |
| net_certainty | mini_budget | -0.0126 | 0.0239 | 0.5972 | 0.9949 | False |
| net_certainty | ukraine_invasion | -0.0005 | 0.0022 | 0.8084 | 0.9949 | False |
| net_certainty | labour_leadership_crisis | 0.0042 | 0.0053 | 0.4268 | 0.9949 | False |

## H3 - party interaction (pooled any_crisis, PM fixed effects, HC3 robust SE)

`DV ~ PM fixed effects + any_crisis + is_labour:any_crisis`. Per-crisis x party interactions are not identifiable - each named crisis window overlaps exactly one party's tenure in this corpus (event_study.py), so those interaction cells are structurally empty. `any_crisis` is the pooled Conservative crisis effect (Conservative is the reference party); `is_labour:any_crisis` is the differential Labour effect - the H3 test. Benjamini-Hochberg FDR correction applied across all 3 tests (1 interaction x 3 dependent variables) together.

**Power caveat**: the Labour side of this interaction is identified from a single crisis window under a single PM (Starmer, Labour leadership crisis); the Conservative side pools 3 windows across 2 PMs (Johnson x2, Truss x1). This asymmetry was flagged in `PHASE0_SCOPING.md` before the corpus was built and is not a defect introduced at this stage - a non-significant or unstable Labour estimate reflects the data available, not a modeling error.

| dv | term | coef | se | pvalue | pvalue_bh | significant_bh |
|---|---|---|---|---|---|---|
| vader_compound | is_labour:any_crisis | -0.0223 | 0.0492 | 0.6503 | 0.9167 | False |
| transformer_score | is_labour:any_crisis | 0.0101 | 0.0969 | 0.9167 | 0.9167 | False |
| net_certainty | is_labour:any_crisis | 0.0058 | 0.0054 | 0.2796 | 0.8387 | False |

## Conclusion

**No effect survives Benjamini-Hochberg correction, for H2 or H3.** This is reported as a genuine null result, not suppressed or re-tested with different windows to find significance.

The closest raw (uncorrected) result is sentiment during Covid-19 (p=0.081 VADER, p=0.068 transformer) - but in the direction opposite H2's prediction: both scores lean slightly *more positive* than Johnson's own baseline, not more negative. Neither survives correction. Ukraine and the Labour leadership crisis show no signal in either direction; mini-budget is uninterpretable (1 sitting). H3's interaction terms are all far from significance.

Plausible reasons this corpus doesn't show the predicted crisis effect, none tested further here per the decision to stop at this window definition:

- VADER and the general-purpose transformer model are tuned for everyday text, not the formal, guarded register of Commons speeches - a PM under pressure may sound measured rather than negative, even in substance.
- Crisis windows spanning weeks to over a year (Covid) average over a lot of non-crisis-toned procedural speech, which can dilute a real but short-lived rhetorical spike.
- H2 itself may simply not hold for this corpus - a null result is not a failure of the analysis, it is the analysis's answer.
