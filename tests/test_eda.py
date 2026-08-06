import pandas as pd

from hansard_pm_nlp.eda import build_mtld_over_time, build_summary


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


def test_build_summary_tfidf_excludes_hansard_address_vocabulary():
    df = pd.DataFrame(
        {
            "pm_name": ["A", "B"],
            "contribution_text": [
                "My hon. Friend is right, hon. Members will know Brexit means Brexit.",
                "The hon. Gentleman and right hon. Friend know Covid restrictions ease.",
            ],
            "is_pmqs": [True, True],
        }
    )
    summary = build_summary(df, top_k=5)
    for _, row in summary.iterrows():
        assert "hon" not in row["top_tfidf_terms"]
        assert "friend" not in row["top_tfidf_terms"]


def _toy_time_df():
    # PM A: 10 words in January (survives a min_words=5 floor).
    # PM B: 3 words in January (dropped by the same floor).
    return pd.DataFrame(
        {
            "pm_name": ["A", "A", "B"],
            "sitting_date": ["2020-01-05", "2020-01-20", "2020-01-10"],
            "contribution_text": [
                "one two three four five",
                "six seven eight nine ten",
                "only three words",
            ],
        }
    )


def test_build_mtld_over_time_drops_sparse_bins():
    result = build_mtld_over_time(_toy_time_df(), min_words=5)
    assert set(result["pm_name"]) == {"A"}


def test_build_mtld_over_time_keeps_word_rich_bins():
    result = build_mtld_over_time(_toy_time_df(), min_words=5)
    row = result.iloc[0]
    assert row["pm_name"] == "A"
    assert row["word_count"] == 10
    assert row["period"] == pd.Timestamp("2020-01-01")


def test_build_mtld_over_time_empty_when_nothing_survives_the_floor():
    result = build_mtld_over_time(_toy_time_df(), min_words=1000)
    assert result.empty
