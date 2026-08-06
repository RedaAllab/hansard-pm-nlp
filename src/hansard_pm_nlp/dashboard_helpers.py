"""Pure data-transformation helpers for the Phase 8 dashboard (app/app.py).

Kept separate from app.py so the transformation logic is unit-testable
without a Streamlit runtime, per CLAUDE.md §6 (reusable logic belongs in a
module, not inlined in the app).
"""

import pandas as pd


def normalize_radar(
    profile: pd.DataFrame, cols: list[str], selected_pms: list[str]
) -> pd.DataFrame:
    """Min-max normalize each stylometric column across only `selected_pms`.

    Scoping the min/max to the current selection - rather than the full
    profile - means deselecting an outlier PM (e.g. Liz Truss, n=123
    contributions vs. 2,195-5,459 for the other three) actually rescales
    the remaining axes instead of leaving
    them compressed by a PM that isn't even drawn anymore. A column where
    every selected PM ties (including the single-PM-selected case, where
    every column trivially ties) would otherwise divide by zero; those are
    centered at 0.5 rather than left as NaN.
    """
    subset = profile.set_index("pm_name").loc[selected_pms, cols]
    span = (subset.max() - subset.min()).replace(0, pd.NA)
    return ((subset - subset.min()) / span).fillna(0.5)


def confusion_cell_detail(
    predictions: pd.DataFrame, model_col: str, actual_pm: str, predicted_pm: str
) -> pd.DataFrame:
    """Sittings behind one confusion-matrix cell (actual PM x predicted PM).

    Backs the classifier tab's click-to-drill-down: the aggregate count in a
    cell doesn't say *which* sittings were confused, this does.
    """
    mask = (predictions["pm_name"] == actual_pm) & (predictions[model_col] == predicted_pm)
    detail = predictions.loc[mask, ["sitting_date"]].sort_values("sitting_date")
    return detail.reset_index(drop=True)


def parse_tfidf_terms(term_string: str) -> list[tuple[str, float]]:
    """Parse eda.py's "term (score); term (score); ..." column format back
    into (term, score) pairs, in the descending-score order eda.py wrote
    them in.
    """
    if pd.isna(term_string) or not term_string:
        return []
    pairs = []
    for item in term_string.split("; "):
        term, _, score = item.rpartition(" (")
        pairs.append((term, float(score.rstrip(")"))))
    return pairs


def crisis_baseline_split(event_df: pd.DataFrame, crisis_col: str, metric_col: str) -> pd.DataFrame:
    """Long-format table splitting `metric_col` into baseline vs. crisis-
    window periods, using the same boolean crisis dummy Phase 7's OLS
    regression tests (event_study.py) - so this box plot illustrates
    exactly what was tested, not a different ad hoc windowing.
    """
    period = event_df[crisis_col].map({True: "Crisis", False: "Baseline"})
    return pd.DataFrame({"period": period, "value": event_df[metric_col]})


def pmqs_split_by_pm(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-PM version of affect.build_pmqs_split: hedging/certainty split by
    debate type (PMQs vs. other) within each PM, to check whether the
    PMQs-hedges-*less* paradox (affect.py, affect_report.md, whole-corpus)
    holds for every PM individually or is driven by one of them.

    Deliberately not calling affect.build_pmqs_split(scored, ...) here even
    though the aggregation is otherwise identical plus a groupby key:
    affect.py imports sentiment.py, which imports vaderSentiment at module
    level - a dependency requirements-app.txt (the dashboard's lightweight
    deploy path) doesn't include, since the dashboard never calls
    sentiment.py/affect.py at runtime (see README's "Dashboard" section).
    Importing affect.py from here would silently pull that dependency back
    into the dashboard's import graph. If affect.build_pmqs_split's
    aggregation ever changes, update this one to match.
    """
    agg = scored.groupby(["pm_name", "is_pmqs"]).agg(
        n_contributions=("contribution_text", "size"),
        mean_hedge_rate=("hedge_rate", "mean"),
        mean_net_certainty=("net_certainty", "mean"),
    )
    return agg.reset_index()


def merge_overlapping_topics(
    topic_df: pd.DataFrame, topic_labels: dict[str, str], merged_label: str
) -> pd.DataFrame:
    """Sum the one documented near-duplicate LDA topic pair (0 and 1, both
    Ukraine/Russia/security - phase5_lda_report.md) into a single column and
    rename the rest via `topic_labels`. Shared by the topics-over-time area
    chart and the mean-topic-weight-by-PM heatmap so both always agree on
    which topics are merged and how they're labeled, instead of each tab
    carrying its own copy of the merge that could drift out of sync.
    """
    merged = topic_df.copy()
    merged[merged_label] = merged["topic_0"] + merged["topic_1"]
    return merged.drop(columns=["topic_0", "topic_1"]).rename(columns=topic_labels)


def crisis_party_split(event_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Long-format table for the H3 pooled any-crisis x party interaction:
    one group per (party, in/out of any named crisis) combination, matching
    the regression's interaction term (per-crisis x party isn't
    identifiable in this corpus - see phase7_event_study_report.md).
    """
    period = event_df["any_crisis"].map({True: "Crisis", False: "Baseline"})
    return pd.DataFrame(
        {"party": event_df["pm_party"], "period": period, "value": event_df[metric_col]}
    )
