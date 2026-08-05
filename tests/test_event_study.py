import pandas as pd

from hansard_pm_nlp.event_study import add_crisis_dummies, build_sitting_dataset


def _toy_contributions():
    long_text = " ".join(["word"] * 60)
    rows = [
        {
            "pm_name": "Boris Johnson",
            "pm_party": "Conservative",
            "sitting_date": pd.Timestamp("2020-04-01"),
            "contribution_text": long_text,
            "is_pmqs": True,
            "vader_compound": -0.5,
            "transformer_score": -0.3,
            "hedge_rate": 0.02,
            "booster_rate": 0.01,
            "net_certainty": -0.01,
        },
        {
            "pm_name": "Boris Johnson",
            "pm_party": "Conservative",
            "sitting_date": pd.Timestamp("2019-08-01"),
            "contribution_text": long_text,
            "is_pmqs": False,
            "vader_compound": 0.2,
            "transformer_score": 0.1,
            "hedge_rate": 0.01,
            "booster_rate": 0.02,
            "net_certainty": 0.01,
        },
        {
            "pm_name": "Keir Starmer",
            "pm_party": "Labour",
            "sitting_date": pd.Timestamp("2026-06-01"),
            "contribution_text": long_text,
            "is_pmqs": True,
            "vader_compound": -0.4,
            "transformer_score": -0.2,
            "hedge_rate": 0.03,
            "booster_rate": 0.0,
            "net_certainty": -0.03,
        },
    ]
    return pd.DataFrame(rows)


def test_build_sitting_dataset_aggregates_one_row_per_pm_and_date():
    docs = build_sitting_dataset(_toy_contributions())
    assert len(docs) == 3
    assert set(docs.columns) >= {"pm_name", "sitting_date", "pm_party", "vader_compound"}


def test_build_sitting_dataset_drops_short_sittings():
    df = _toy_contributions()
    df.loc[0, "contribution_text"] = "one word"
    docs = build_sitting_dataset(df)
    assert len(docs) == 2


def test_add_crisis_dummies_flags_covid_window():
    docs = build_sitting_dataset(_toy_contributions())
    docs = add_crisis_dummies(docs)
    johnson_covid_row = docs[(docs["pm_name"] == "Boris Johnson") & (docs["sitting_date"] == "2020-04-01")]
    assert johnson_covid_row["crisis_covid19"].iloc[0]
    assert johnson_covid_row["any_crisis"].iloc[0]


def test_add_crisis_dummies_false_outside_any_window():
    docs = build_sitting_dataset(_toy_contributions())
    docs = add_crisis_dummies(docs)
    baseline_row = docs[docs["sitting_date"] == "2019-08-01"]
    assert not baseline_row["any_crisis"].iloc[0]
    assert not baseline_row["crisis_covid19"].iloc[0]


def test_add_crisis_dummies_flags_labour_leadership_crisis():
    docs = build_sitting_dataset(_toy_contributions())
    docs = add_crisis_dummies(docs)
    starmer_row = docs[docs["pm_name"] == "Keir Starmer"]
    assert starmer_row["crisis_labour_leadership_crisis"].iloc[0]
