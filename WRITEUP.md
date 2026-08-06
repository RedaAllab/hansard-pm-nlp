# UK Prime Ministers' Rhetoric, 2019-2026: Methods and Results

A companion to the [README](README.md), for a reader who wants the reasoning behind each modeling choice and the full results against each pre-registered hypothesis, including the ones that came back null. Written for a technical but non-specialist audience: no assumed background in NLP or econometrics beyond what's defined inline.

Full per-phase detail, tables, and generation timestamps live in `data/processed/*.md`; this document synthesizes them into one narrative rather than repeating them.

## 1. Motivation and research question

Four Prime Ministers held the office between 2019 and the corpus cutoff: Boris Johnson, Liz Truss, Rishi Sunak, and Keir Starmer, spanning Brexit's final stretch, Covid-19, the 2022 mini-budget crisis, the invasion of Ukraine, and the run-up to a mid-2026 change of Labour leadership. That density of exogenous shocks, packed into under seven years, makes the period an unusually good natural experiment for studying political rhetoric under pressure.

The analysis asks two linked questions, treated as a quasi-experimental event study rather than an open-ended exploration:

1. Does each PM have a measurably distinct rhetorical signature (**H1**)?
2. Does rhetoric shift in sentiment and certainty around major crises, and does that shift depend on the governing party (**H2**, **H3**)? A fourth, exploratory question (**H4**) asks whether topic emphasis drifts continuously or breaks sharply at PM transitions.

Hypotheses and crisis-window boundaries were fixed in [`hansard-pm-extraction`'s `PHASE0_SCOPING.md`](https://github.com/RedaAllab/hansard-pm-extraction/blob/main/PHASE0_SCOPING.md) before any modeling began, including the statistical-power caveat for H3 (below) - the corpus was not adjusted after the fact to chase significance.

## 2. Data

10,673 Commons Spoken contributions by the sitting PM, fetched from the UK Parliament Hansard API and joined against PM identity via a **temporal join** against tenure windows resolved from the Members API's `governmentPosts` endpoint - never a static date-range lookup, since a static join would misattribute speeches around transition dates. 6,076 of the 10,673 contributions (57%) are PMQs ("Engagements" in Hansard's own filing); the rest span named debates (Covid-19 Update, Ukraine, Middle East, Sue Gray Report, and others).

| PM | Party | Tenure | Contributions | Words | PMQs share |
|---|---|---|---|---|---|
| Boris Johnson | Conservative | 2019-07-24 to 2022-09-06 | 5,459 | 520,674 | 49% |
| Liz Truss | Conservative | 2022-09-06 to 2022-10-25 | 123 | 8,842 | 79% |
| Rishi Sunak | Conservative | 2022-10-25 to 2024-07-05 | 2,195 | 222,152 | 67% |
| Keir Starmer | Labour | 2024-07-05 to 2026-07-20 | 2,896 | 284,990 | 63% |

Truss's 49-day tenure leaves her with roughly a seventeenth to a forty-fourth as much text as the other three - every downstream analysis that compares her to the rest flags this explicitly rather than treating her numbers as equally reliable.

For topic modeling, style classification, and the event study, contributions are aggregated to one document per (PM, sitting date) - 296 documents after dropping 2 with fewer than 50 words. Raw contributions range from one-word interjections ("No.") to multi-page statements, too uneven for stable per-document statistics at the individual-contribution level.

## 3. Methods

**Lexical baseline.** Readability (Flesch-Kincaid grade) and sentence length are averaged per contribution, so one long statement can't dominate a PM's score. Lexical diversity uses MTLD (McCarthy & Jarvis, 2010) rather than the more common type-token ratio (TTR): TTR is length-biased, and with corpus sizes ranging from 8,842 to 520,674 words across PMs, a raw TTR comparison would mostly measure corpus size. Distinctive vocabulary per PM comes from TF-IDF over one concatenated document per PM, with Hansard's parliamentary-address vocabulary ("hon", "right", "friend", "gentleman") filtered out via a curated stopword list - otherwise it dominates every PM's chart identically and says nothing PM-specific.

**Sentiment and hedging.** Two independent sentiment measures are compared rather than trusted singly: VADER (a lexicon-based baseline) and a pretrained transformer (`distilbert-base-uncased-finetuned-sst-2-english`). The two agree on positive-vs-negative sign 74-79% of the time depending on PM - real but not total agreement, consistent with them capturing overlapping but distinct signals. A custom lexicon of modal verbs, hedge verbs, and intensity adverbs produces a hedging rate and a "net certainty" score (boosting language minus hedging language), aimed specifically at PMQs, a genre built around evasiveness.

**Topic modeling.** LDA (gensim, K=14) and BERTopic were compared explicitly rather than picking one on assumption. K was chosen by hand after a coherence sweep across K=5-30 found no clean plateau, then refined after adding a second stopword pass (`FILLER_STOPWORDS`: "thank", "important", "grateful", and similar politeness formulas that showed up as noise in the first run's topic word lists) - final coherence (c_v) 0.554. BERTopic's default UMAP+HDBSCAN pipeline was also tried at two settings; both collapsed most of the 296-document corpus into one or two broad clusters (61-82% of documents in a single catch-all topic) rather than separating cleanly. This is a corpus-size limitation of BERTopic's default clustering - it's designed for, and typically evaluated on, corpora several orders of magnitude larger than 296 documents - not evidence that LDA's topics are objectively better. LDA was used going forward. Two LDA topics (both Ukraine/Russia/security, at every K value tested) overlap by design and are read as a real structural feature of the corpus - PMs discuss Russia/Ukraine across genuinely distinct sub-periods with different vocabulary each time - rather than a preprocessing artifact to merge away.

**Style classification (H1).** Style features (lexical diversity, readability, hedging/certainty rates, individual function-word frequencies, POS-tag distribution via spaCy) feed a logistic regression and a HistGradientBoosting classifier, both with `class_weight='balanced'` to handle the class imbalance (Johnson 144 documents, Sunak 63, Starmer 84, pre-split). Liz Truss is excluded from this analysis specifically - 5 documents is too few for a train/test split or reliable per-class metrics, the same precedent later applied to Andy Burnham's still-thin sample. The train/test split is **temporal** (earlier sittings train, later sittings test, 232/59 documents), not random - a random split would let the model partly memorize a shared news cycle rather than testing genuine generalization to a PM's later style.

**Event-study statistics (H2, H3).** OLS with PM fixed effects and one boolean dummy per named crisis window, robust (HC3) standard errors, and Benjamini-Hochberg false-discovery-rate correction applied across the full family of tests together (12 tests for H2: 4 crises x 3 outcome variables; 3 tests for H3). Per-crisis x party interactions are not identifiable in this corpus - each named crisis window overlaps exactly one governing party's tenure - so H3 instead tests a single pooled `any_crisis x is_labour` interaction term.

## 4. Results by hypothesis

**H1 - stylometric signature: confirmed.** Both classifiers attribute held-out, later-tenure speech excerpts to the correct PM well above chance: 91.5% accuracy (logistic regression) and 93.2% (HistGradientBoosting), against chance baselines of 33.3% (uniform over 3 classes) and 49.2% (always predict the majority class). The confusion matrices are close to diagonal for both models - only 5 and 4 misclassifications respectively out of 59 test documents, with no single pair of PMs dominating both models' errors (logistic regression's errors concentrate on Johnson/Starmer; HistGradientBoosting's skew more toward Johnson/Sunak instead). Permutation feature importance tells a genre-consistent story: for the linear model, individual function words dominate (auxiliary "had"/"been"/"was", the negator "not"); for the boosted-tree model, `hedge_rate` and interjection frequency (`pos_INTJ`) rank in the top three - stylometric identity here is carried more by function-word rhythm and PMQs-style delivery tics than by topic vocabulary, which the features deliberately exclude.

**H2 - crisis affect: null result.** No crisis-window coefficient survives Benjamini-Hochberg correction, for any of the three outcome variables (VADER sentiment, transformer sentiment, net certainty). The closest raw (uncorrected) signal is Covid-19 sentiment (p=0.081 VADER, p=0.068 transformer) - but pointed in the *opposite* direction to the hypothesis: both scores lean slightly more positive during Covid-19 than each PM's own baseline, not more negative. The mini-budget window is a special case worth flagging rather than hiding: it contains exactly one sitting (Liz Truss, 2022-10-12), so its standard error (~3.0, on a score bounded in [-1, 1]) reflects single-observation identification, not a genuine null - that row is uninterpretable given available data, not evidence against H2 for that crisis specifically.

**H3 - party interaction: null result, and underpowered by construction.** The Labour-differential interaction term is small and non-significant across all three outcome variables. This was flagged as a live risk before the corpus was even built (`hansard-pm-extraction`'s `PHASE0_SCOPING.md`): the Labour side of the interaction is identified from a single crisis window under a single PM (Starmer, the leadership crisis), while the Conservative side pools three windows across two PMs. A non-significant or unstable Labour estimate reflects the data actually available in this seven-year window, not a modeling defect - re-testing this properly needs either more Labour-tenure crisis windows or a differently scoped corpus, not a different specification of the same regression.

**H4 - thematic drift (exploratory): descriptive, not formally tested.** The dashboard's Topics tab shows both continuous drift within a tenure (e.g. Covid-related topics moving from restrictions/testing to vaccines/schools to NHS pay as the pandemic's phases changed) and visible breaks aligned with PM transitions, but this hypothesis was scoped as exploratory from the start (`hansard-pm-extraction`'s `CLAUDE.md` §2) and isn't run through a formal break-point test here - see the dashboard for the visual evidence.

## 5. Limitations

Stated plainly rather than left for a reader to discover:

- **Truss's 49-day tenure** makes her the thinnest slice of every per-PM comparison in this corpus (8,842 words vs. 220k-520k for the other three); she's excluded outright from the classifier and flagged everywhere else she appears.
- **General-purpose sentiment models on a formal register.** VADER and the transformer model are both tuned for everyday text, not the guarded, procedural register of Commons speeches. A PM under real pressure may read as measured rather than negative in substance, which is one plausible reason H2 didn't find the predicted effect.
- **BERTopic needs a bigger corpus than this one has** to cluster meaningfully (see Section 3) - a limitation of the corpus size relative to that specific method's assumptions, not a general indictment of transformer-based topic modeling.
- **H3's asymmetric power** between the two parties is a structural feature of a seven-year window that happens to contain far more Conservative than Labour crisis-adjacent tenure, not something a different regression specification could fix.
- **Crisis windows are fixed, dated choices** (`hansard-pm-extraction`'s `PHASE0_SCOPING.md`), not re-tuned after seeing results - the Covid-19 window, for instance, spans 16 months and likely averages a lot of non-crisis-toned procedural speech alongside any real spike, which could dilute a short-lived rhetorical effect into statistical noise.

## 6. Reproducibility

Every stage is a standalone module (`python -m hansard_pm_nlp.<module>`), writing to `data/processed/` alongside the markdown report it was generated from, with a timestamp in the report itself. Random seeds are fixed for the LDA run (seed=42) and the classifiers; the train/test split is temporal and deterministic rather than a random seed that would need documenting. The corpus cutoff date is recorded once in `data_README.md` at extraction time and is not silently advanced - extending the corpus (for instance, once Andy Burnham's tenure has produced enough Hansard data to analyze) is a deliberate, separately documented re-run, not an automatic background refresh.

## 7. Conclusion

Two of four hypotheses hold up, and two don't - and the two that don't are reported with the same rigor as the two that do. PMs in this corpus are stylometrically distinguishable well above chance (H1), largely through function-word and delivery patterns rather than topic choice. Crisis windows, at least as defined here with two general-purpose sentiment models, don't show the predicted negative-sentiment shift (H2), and the party-interaction test (H3) is honestly underpowered by the shape of the seven-year window studied rather than by a flaw in the regression. That asymmetry between what worked and what didn't is the more interesting finding than a clean four-for-four would have been: it says something specific about what off-the-shelf sentiment tooling can and can't detect in a formal, high-stakes register, not just about these four Prime Ministers.
