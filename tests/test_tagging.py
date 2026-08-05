import pandas as pd

from hansard_pm_nlp.tagging import add_is_pmqs


def test_flags_engagements_as_pmqs():
    df = pd.DataFrame({"debate_section": ["Engagements", "Ukraine"]})
    result = add_is_pmqs(df)
    assert result["is_pmqs"].tolist() == [True, False]


def test_does_not_mutate_input():
    df = pd.DataFrame({"debate_section": ["Engagements"]})
    add_is_pmqs(df)
    assert "is_pmqs" not in df.columns


def test_requires_pre_normalized_debate_section():
    df = pd.DataFrame({"debate_section": [" Engagements"]})
    result = add_is_pmqs(df)
    assert result["is_pmqs"].tolist() == [False]
