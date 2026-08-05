"""Debate-type tagging: flag PMQs contributions.

Prime Minister's Questions is filed under the debate title "Engagements" in
Hansard, not a dedicated PMQs field - this is the API's own naming, confirmed
by inspecting the corpus (see PROJECT_SUMMARY.md §3, PMQs vs other debates).
"""

import pandas as pd

PMQS_DEBATE_SECTION = "Engagements"


def add_is_pmqs(df: pd.DataFrame, debate_section_col: str = "debate_section") -> pd.DataFrame:
    """Add an `is_pmqs` boolean column. Expects `debate_section_col` already
    whitespace-normalized (see cleaning.clean_debate_section).
    """
    df = df.copy()
    df["is_pmqs"] = df[debate_section_col] == PMQS_DEBATE_SECTION
    return df
