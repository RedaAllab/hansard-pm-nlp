import pandas as pd

from hansard_pm_nlp.eda import build_summary


def _toy_df():
    return pd.DataFrame(
        {
            "pm_name": ["A", "A", "B"],
            "contribution_text": [
                "Brexit means Brexit. We will deliver Brexit for the British people.",
                "The economy is strong and getting stronger every single day.",
                "Covid restrictions will ease as soon as it is safe to do so.",
            ],
            "is_pmqs": [True, False, True],
        }
    )


def test_build_summary_has_one_row_per_pm():
    summary = build_summary(_toy_df(), top_k=5)
    assert set(summary["pm_name"]) == {"A", "B"}


def test_build_summary_counts_contributions_correctly():
    summary = build_summary(_toy_df(), top_k=5)
    row_a = summary[summary["pm_name"] == "A"].iloc[0]
    assert row_a["n_contributions"] == 2


def test_build_summary_pmqs_share():
    summary = build_summary(_toy_df(), top_k=5)
    row_a = summary[summary["pm_name"] == "A"].iloc[0]
    assert row_a["pmqs_share"] == 0.5


def test_build_summary_surfaces_distinctive_terms():
    summary = build_summary(_toy_df(), top_k=5)
    row_a = summary[summary["pm_name"] == "A"].iloc[0]
    row_b = summary[summary["pm_name"] == "B"].iloc[0]
    assert "brexit" in row_a["top_tfidf_terms"]
    assert "covid" in row_b["top_tfidf_terms"]
