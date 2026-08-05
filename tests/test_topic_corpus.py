import pandas as pd

from hansard_pm_nlp.topic_corpus import build_topic_documents


def test_build_topic_documents_groups_by_pm_and_date():
    df = pd.DataFrame(
        {
            "pm_name": ["A", "A", "B"],
            "sitting_date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01"]),
            "contribution_text": [
                " ".join(["word"] * 60),
                " ".join(["word"] * 60),
                " ".join(["word"] * 60),
            ],
            "is_pmqs": [True, False, True],
        }
    )
    docs = build_topic_documents(df)
    assert len(docs) == 2
    row_a = docs[docs["pm_name"] == "A"].iloc[0]
    assert row_a["n_contributions"] == 2
    assert row_a["word_count"] == 120


def test_build_topic_documents_drops_short_docs():
    df = pd.DataFrame(
        {
            "pm_name": ["A", "B"],
            "sitting_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "contribution_text": [" ".join(["word"] * 60), "too short"],
            "is_pmqs": [True, False],
        }
    )
    docs = build_topic_documents(df)
    assert len(docs) == 1
    assert docs.iloc[0]["pm_name"] == "A"


def test_build_topic_documents_pmqs_share():
    df = pd.DataFrame(
        {
            "pm_name": ["A", "A"],
            "sitting_date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "contribution_text": [" ".join(["word"] * 60)] * 2,
            "is_pmqs": [True, False],
        }
    )
    docs = build_topic_documents(df)
    assert docs.iloc[0]["pmqs_share"] == 0.5
