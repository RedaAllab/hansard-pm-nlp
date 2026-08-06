# Phase 3 - Lexical baseline

Generated: 2026-08-06T07:23:49.770172+00:00

Descriptive statistics only - no hypothesis tests, no modeling. Readability/sentence length are averaged per contribution; lexical diversity (TTR, MTLD) and n-grams are computed on each PM's full concatenated corpus (see hansard_pm_nlp.eda docstring for why).

`mtld_over_time.parquet` additionally bins MTLD by month per PM (build_mtld_over_time()), dropping any bin under 1,500 words so a sparse month doesn't get plotted as a misleadingly precise point - see the dashboard's Overview tab.

## Boris Johnson

- Contributions: 5459 (520674 words, 49% PMQs)
- Type-token ratio: 0.0260 (biased by corpus size - see MTLD instead for cross-PM comparison)
- MTLD: 76.8
- Mean Flesch-Kincaid grade: 9.42
- Mean words per sentence: 19.7
- Top bigrams: of the (3829); we are (2664); in the (2340); that we (2326); that is (2248); right hon (2240); hon friend (2192); we have (1988); it is (1929); to the (1889); we will (1740); the right (1674); i am (1617); that the (1582); my hon (1468)
- Top TF-IDF terms (distinctive vs. other PMs): people (0.367); country (0.351); think (0.212); government (0.192); way (0.150); support (0.149); uk (0.133); want (0.131); say (0.118); going (0.111); said (0.109); course (0.108); know (0.108); doing (0.103); just (0.102)

## Keir Starmer

- Contributions: 2896 (284990 words, 63% PMQs)
- Type-token ratio: 0.0312 (biased by corpus size - see MTLD instead for cross-PM comparison)
- MTLD: 77.7
- Mean Flesch-Kincaid grade: 8.96
- Mean words per sentence: 18.1
- Top bigrams: of the (1659); we are (1571); it is (1303); that is (1294); in the (1238); that we (1160); we have (1153); to the (883); hon friend (879); we will (872); my hon (841); i am (794); that the (754); the right (712); and i (696)
- Top TF-IDF terms (distinctive vs. other PMs): government (0.204); people (0.179); work (0.171); country (0.169); years (0.132); support (0.131); security (0.130); issue (0.127); need (0.122); know (0.122); defence (0.118); make (0.116); working (0.113); opposition (0.111); did (0.108)

## Liz Truss

- Contributions: 123 (8842 words, 79% PMQs)
- Type-token ratio: 0.1663 (biased by corpus size - see MTLD instead for cross-PM comparison)
- MTLD: 76.4
- Mean Flesch-Kincaid grade: 9.42
- Mean words per sentence: 18.2
- Top bigrams: we are (70); that we (60); we will (49); i am (48); right hon (43); hon friend (42); that is (39); sure that (39); make sure (35); we have (34); with the (34); of the (33); in the (30); to make (28); the right (27)
- Top TF-IDF terms (distinctive vs. other PMs): energy (0.440); people (0.334); sure (0.266); make (0.208); need (0.155); support (0.135); way (0.131); country (0.131); new (0.126); price (0.106); secretary (0.106); growth (0.102); economy (0.097); help (0.097); work (0.092)

## Rishi Sunak

- Contributions: 2195 (222152 words, 67% PMQs)
- Type-token ratio: 0.0364 (biased by corpus size - see MTLD instead for cross-PM comparison)
- MTLD: 87.4
- Mean Flesch-Kincaid grade: 10.74
- Mean words per sentence: 20.9
- Top bigrams: of the (1094); that is (1091); we are (1089); that we (1056); it is (1037); hon friend (1020); in the (966); we have (939); right hon (755); to the (731); my hon (712); that the (644); we will (633); the right (589); i am (583)
- Top TF-IDF terms (distinctive vs. other PMs): people (0.297); support (0.211); government (0.184); know (0.155); new (0.150); country (0.140); ensure (0.131); just (0.127); work (0.122); labour (0.119); said (0.119); make (0.112); continue (0.112); uk (0.112); year (0.110)

