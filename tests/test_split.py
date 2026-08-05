import pandas as pd

from hansard_pm_nlp.split import temporal_train_test_split


def _toy_docs():
    rows = []
    for pm, n in [("Boris Johnson", 10), ("Rishi Sunak", 10), ("Liz Truss", 5)]:
        for i in range(n):
            date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
            rows.append({"pm_name": pm, "sitting_date": date})
    return pd.DataFrame(rows)


def test_excludes_liz_truss():
    train, test = temporal_train_test_split(_toy_docs())
    assert "Liz Truss" not in train["pm_name"].values
    assert "Liz Truss" not in test["pm_name"].values


def test_test_set_is_always_later_than_train_per_pm():
    train, test = temporal_train_test_split(_toy_docs(), test_size=0.2)
    for pm in ["Boris Johnson", "Rishi Sunak"]:
        max_train_date = train.loc[train["pm_name"] == pm, "sitting_date"].max()
        min_test_date = test.loc[test["pm_name"] == pm, "sitting_date"].min()
        assert max_train_date < min_test_date


def test_every_remaining_pm_has_at_least_one_test_doc():
    train, test = temporal_train_test_split(_toy_docs(), test_size=0.2)
    assert set(test["pm_name"].unique()) == {"Boris Johnson", "Rishi Sunak"}
