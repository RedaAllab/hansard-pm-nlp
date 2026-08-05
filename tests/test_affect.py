import pandas as pd

from hansard_pm_nlp.affect import build_pm_summary, build_pmqs_split, score_corpus


def _toy_df():
    return pd.DataFrame(
        {
            "pm_name": ["A", "A", "B"],
            "contribution_text": [
                "This is a wonderful and fantastic achievement for the country.",
                "It might possibly work, but we could be wrong.",
                "This is a disastrous and terrible failure.",
            ],
            "is_pmqs": [True, False, True],
        }
    )


def test_score_corpus_adds_expected_columns():
    scored = score_corpus(_toy_df())
    for col in [
        "vader_compound",
        "transformer_score",
        "hedge_rate",
        "booster_rate",
        "net_certainty",
        "sentiment_sign_agree",
    ]:
        assert col in scored.columns


def test_build_pm_summary_one_row_per_pm():
    scored = score_corpus(_toy_df())
    summary = build_pm_summary(scored)
    assert set(summary["pm_name"]) == {"A", "B"}


def test_build_pm_summary_hedging_reflects_toy_text():
    scored = score_corpus(_toy_df())
    summary = build_pm_summary(scored)
    row_a = summary[summary["pm_name"] == "A"].iloc[0]
    assert row_a["mean_hedge_rate"] > 0


def test_build_pmqs_split_has_both_groups():
    scored = score_corpus(_toy_df())
    split = build_pmqs_split(scored)
    assert set(split["is_pmqs"]) == {True, False}
