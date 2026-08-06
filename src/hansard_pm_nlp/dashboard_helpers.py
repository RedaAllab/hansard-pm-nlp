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
    profile - means deselecting an outlier PM (e.g. Liz Truss, n=5
    contributions) actually rescales the remaining axes instead of leaving
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
    if not term_string:
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
