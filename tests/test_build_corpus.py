import pandas as pd

from hansard_pm_nlp.build_corpus import build_processed_contributions


def test_build_processed_contributions_cleans_and_tags(tmp_path):
    input_path = tmp_path / "pm_contributions.parquet"
    raw = pd.DataFrame(
        {
            "contribution_text": ["Hello <em>world</em>.", "Plain text."],
            "debate_section": [" Engagements", "Ukraine "],
        }
    )
    raw.to_parquet(input_path, index=False)

    df = build_processed_contributions(input_path)

    assert df["contribution_text_raw"].tolist() == ["Hello <em>world</em>.", "Plain text."]
    assert df["contribution_text"].tolist() == ["Hello world.", "Plain text."]
    assert df["debate_section"].tolist() == ["Engagements", "Ukraine"]
    assert df["is_pmqs"].tolist() == [True, False]
