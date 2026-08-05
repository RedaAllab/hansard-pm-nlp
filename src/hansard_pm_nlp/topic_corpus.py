"""Aggregate contributions into one document per PM x sitting date.

Contribution-level documents range from one-word interjections to
multi-page statements, too uneven for LDA/BERTopic to produce stable
per-document topic distributions. Aggregating by PM x sitting date gives
298 documents (confirmed by inspection), median ~2,700 words, only 2 under
100 words - those 2 are dropped rather than modeled, since a handful of
words can't support a meaningful topic distribution either way.
"""

import pandas as pd

MIN_DOC_WORDS = 50


def build_topic_documents(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (pm_name, sitting_date): concatenated text plus metadata
    needed for later joins (contribution count, PMQs share).
    """
    grouped = df.groupby(["pm_name", "sitting_date"])
    docs = grouped.agg(
        text=("contribution_text", lambda s: " ".join(s)),
        n_contributions=("contribution_text", "size"),
        pmqs_share=("is_pmqs", "mean"),
    ).reset_index()
    docs["word_count"] = docs["text"].str.split().str.len()

    dropped = docs[docs["word_count"] < MIN_DOC_WORDS]
    if len(dropped) > 0:
        for _, row in dropped.iterrows():
            print(
                f"Dropping {row['pm_name']} / {row['sitting_date'].date()}: "
                f"{row['word_count']} words (< {MIN_DOC_WORDS})"
            )
    docs = docs[docs["word_count"] >= MIN_DOC_WORDS].reset_index(drop=True)
    return docs
