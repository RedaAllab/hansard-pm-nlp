"""Train/test split for the PM-attribution classifier (Phase 6).

A random per-document split would let the classifier learn the vocabulary of
a specific news cycle rather than genuine style, since documents close in
time share topical vocabulary. Splitting by date within each PM's tenure -
train on earlier sittings, test on later ones - tests whether the model
generalizes across periods, which is what H1 actually claims.

Liz Truss (5 documents, 49-day tenure) is excluded: too few documents for a
train/test split or class-level metrics to mean anything.
"""

import pandas as pd

EXCLUDED_PMS = frozenset({"Liz Truss"})


def temporal_train_test_split(
    docs: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per PM, sort by sitting_date and take the last `test_size` fraction as
    test. Excludes EXCLUDED_PMS first, then splits each remaining PM's
    documents independently so every class is represented on both sides.
    """
    docs = docs[~docs["pm_name"].isin(EXCLUDED_PMS)]

    train_parts, test_parts = [], []
    for _, group in docs.groupby("pm_name"):
        group = group.sort_values("sitting_date")
        n_test = max(1, round(len(group) * test_size))
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, test
