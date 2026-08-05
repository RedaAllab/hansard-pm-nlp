# hansard-pm-nlp

NLP analysis of UK Prime Ministers' rhetoric (2019-present), built on the parquet corpus produced by [`hansard-pm-extraction`](https://github.com/RedaAllab/hansard-pm-extraction). Second stage of a two-repository project; see that repo's `PROJECT_SUMMARY.md` for the full project (research question, hypotheses, roadmap) and `CLAUDE.md` for conventions.

## Pipeline

Each stage is a standalone module, runnable independently, writing its output to `data/processed/` alongside a markdown report.

| Phase | Module | Report |
|---|---|---|
| 2. Corpus construction | `build_corpus.py` (cleaning, debate-type tagging) | `data_README.md` |
| 3. Lexical baseline (TF-IDF, readability, lexical diversity) | `eda.py` | `eda_report.md` |
| 4. Sentiment / hedging | `affect.py` (VADER vs. transformer, custom hedging lexicon) | `affect_report.md` |
| 5. Topic modeling | `build_lda_topics.py` (LDA), `build_topic_comparison.py` (vs. BERTopic) | `phase5_lda_report.md`, `phase5_topic_comparison_report.md` |
| 6. PM-attribution classifier (H1) | `build_classifier.py` (logistic regression, HistGradientBoosting) | `phase6_classifier_report.md` |
| 7. Event-study regressions (H2, H3) | `build_event_study.py` (OLS, PM fixed effects, BH correction) | `phase7_event_study_report.md` |
| 8. Dashboard | `app/app.py` (Streamlit) | - |

Run any stage with `python -m hansard_pm_nlp.<module>`, e.g.:

```bash
python -m hansard_pm_nlp.build_corpus
python -m hansard_pm_nlp.eda
python -m hansard_pm_nlp.affect
python -m hansard_pm_nlp.build_lda_topics
python -m hansard_pm_nlp.build_classifier
python -m hansard_pm_nlp.build_event_study
```

Later stages depend on earlier stages' output already existing in `data/processed/`.

## Dashboard

```bash
streamlit run app/app.py
```

Four tabs: stylometric profile by PM, sentiment/certainty over time (with crisis windows), LDA topics over time, and the PM-attribution classifier's results. Filters (PM, date range) apply per tab, scoped to what the underlying data supports - see the in-app captions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

Requires the corpus from `hansard-pm-extraction` in `data/input/` - see `data/input/README.md` for provenance.

## Tests

```bash
pytest
ruff check .
```

## Known limitations

Documented in the reports rather than hidden:

- **Topic modeling**: LDA (K=14) chosen over BERTopic, whose default UMAP+HDBSCAN pipeline collapses this 296-document corpus into 1-2 broad clusters - a corpus-size limitation, not evidence LDA's topics are objectively better (`phase5_topic_comparison_report.md`). LDA topics 0 and 1 (both Ukraine/Russia/security) overlap by design, read as a real structural feature of the corpus rather than a preprocessing artifact.
- **Classifier (H1)**: Liz Truss excluded (5 documents, 49-day tenure - too few for reliable per-class metrics). Train/test split is temporal (by date within each PM's tenure), not random, so the test score reflects generalization to later sittings, not a leaked news cycle.
- **Event study (H2, H3)**: no effect survives Benjamini-Hochberg correction for either hypothesis - reported as a genuine null result. Per-crisis x party interactions are not identifiable (each named crisis window overlaps exactly one party's tenure in this corpus); H3 instead tests a pooled any-crisis x party interaction.
