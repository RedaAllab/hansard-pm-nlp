"""Run the LDA coherence (c_v) sweep over K and save results for review.

This is a checkpoint, not a final answer: the resulting table/plot is meant
to be read by a person before picking K, per CLAUDE.md §9 - the script
deliberately does not auto-select the best K.

Usage:
    python -m hansard_pm_nlp.run_coherence_sweep
"""

from pathlib import Path

import pandas as pd

from hansard_pm_nlp.lda import coherence_sweep
from hansard_pm_nlp.preprocessing import apply_bigrams, build_bigram_phraser, tokenize_for_topics
from hansard_pm_nlp.topic_corpus import build_topic_documents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

K_GRID = [5, 8, 10, 12, 15, 18, 20, 25, 30]


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "pm_contributions_clean.parquet")
    docs = build_topic_documents(df)

    tokenized = [tokenize_for_topics(t) for t in docs["text"]]
    phraser = build_bigram_phraser(tokenized)
    tokenized = apply_bigrams(tokenized, phraser)

    result = coherence_sweep(tokenized, k_grid=K_GRID)
    result.to_csv(PROCESSED_DIR / "lda_coherence_sweep.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
