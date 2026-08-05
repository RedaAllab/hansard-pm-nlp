"""Event-study dataset construction (Phase 7, H2/H3): aggregate affect
scores to one row per (PM, sitting date) - the same unit used in Phases 5-6 -
and flag crisis-window membership.

Unlike split.py (Phase 6), Liz Truss is kept here: she is the only PM whose
tenure overlaps the mini-budget crisis window, so dropping her would make
that crisis untestable. Her 5 sittings are well above MIN_DOC_WORDS (844-2142
words each), so there is no thin-data reason to exclude her from this layer.

Crisis windows are dated and justified in PHASE0_SCOPING.md. Each window
falls entirely within one party's tenure (Covid/Ukraine/mini-budget under
Conservative PMs, the Labour leadership crisis under Starmer) - a
crisis-window x party interaction is therefore not identifiable per window
(three of the eight cells are structurally empty). H3 is instead tested via
a pooled any_crisis x party interaction (see regression.py).
"""

import pandas as pd

MIN_DOC_WORDS = 50

CRISIS_WINDOWS = {
    "covid19": ("2020-03-23", "2021-07-19"),
    "mini_budget": ("2022-09-23", "2022-10-17"),
    "ukraine_invasion": ("2022-02-24", "2022-05-24"),
    "labour_leadership_crisis": ("2026-05-07", "2026-07-20"),
}

AFFECT_COLUMNS = [
    "vader_compound",
    "transformer_score",
    "hedge_rate",
    "booster_rate",
    "net_certainty",
]


def build_sitting_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (pm_name, sitting_date): mean affect scores plus
    pm_party, n_contributions, and pmqs_share, dropping sittings under
    MIN_DOC_WORDS (mirrors topic_corpus.build_topic_documents).
    """
    grouped = df.groupby(["pm_name", "sitting_date"])
    agg = {col: (col, "mean") for col in AFFECT_COLUMNS}
    docs = grouped.agg(
        pm_party=("pm_party", "first"),
        n_contributions=("contribution_text", "size"),
        pmqs_share=("is_pmqs", "mean"),
        word_count=("contribution_text", lambda s: sum(len(t.split()) for t in s)),
        **agg,
    ).reset_index()

    docs = docs[docs["word_count"] >= MIN_DOC_WORDS].reset_index(drop=True)
    return docs


def add_crisis_dummies(docs: pd.DataFrame) -> pd.DataFrame:
    """Add one boolean column per named crisis window plus a pooled
    'any_crisis' column (True if the sitting falls in any window).
    """
    docs = docs.copy()
    any_crisis = pd.Series(False, index=docs.index)
    for name, (start, end) in CRISIS_WINDOWS.items():
        in_window = (docs["sitting_date"] >= start) & (docs["sitting_date"] <= end)
        docs[f"crisis_{name}"] = in_window
        any_crisis |= in_window
    docs["any_crisis"] = any_crisis
    return docs
