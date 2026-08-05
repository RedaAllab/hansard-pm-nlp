"""Run BERTopic once more (min_topic_size=10) to export its topics alongside
LDA's, and write the final LDA vs BERTopic comparison - the "final choice
justified in the write-up" PROJECT_SUMMARY.md asks for.

Usage:
    python -m hansard_pm_nlp.build_topic_comparison
"""

import datetime as dt
from pathlib import Path

import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel

from hansard_pm_nlp.bertopic_model import fit_bertopic, get_topic_words
from hansard_pm_nlp.preprocessing import apply_bigrams, build_bigram_phraser, tokenize_for_topics
from hansard_pm_nlp.topic_corpus import build_topic_documents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LDA_COHERENCE = 0.554
MIN_TOPIC_SIZE = 10


def write_report(
    bertopic_words: dict[int, list[str]],
    bertopic_coherence: float,
    topic_sizes: pd.Series,
    n_outliers: int,
    n_docs: int,
    path: Path,
) -> None:
    generated_at = dt.datetime.now(dt.UTC).isoformat()
    lines = [
        "# Phase 5 - LDA vs BERTopic comparison",
        "",
        f"Generated: {generated_at}",
        "",
        "Same 296 documents (PM x sitting date), same STOPWORDS list applied "
        "to keyword extraction in both methods (c-TF-IDF for BERTopic, "
        "dictionary filtering for LDA) - see phase5_lda_report.md for the "
        "LDA side.",
        "",
        "## Coherence (c_v, comparable metric across both)",
        "",
        "| Method | Topics | c_v coherence |",
        "|---|---|---|",
        f"| LDA (K=14) | 14 | {LDA_COHERENCE:.3f} |",
        f"| BERTopic (min_topic_size={MIN_TOPIC_SIZE}) | {len(bertopic_words)} | "
        f"{bertopic_coherence:.3f} |",
        "",
        "Coherence alone understates the gap: BERTopic's number (0.468) isn't "
        "far below LDA's (0.554), but that number doesn't capture that "
        f"{topic_sizes.iloc[0]}/{n_docs} documents ({topic_sizes.iloc[0] / n_docs:.0%}) "
        "land in a single catch-all topic - see below.",
        "",
        "## BERTopic topic sizes",
        "",
        "| Topic | N docs |",
        "|---|---|",
    ]
    for topic_id, count in topic_sizes.items():
        lines.append(f"| {topic_id} | {count} |")
    lines.append(f"| -1 (outliers) | {n_outliers} |")

    lines += ["", "## BERTopic topics", ""]
    for tid, words in bertopic_words.items():
        lines.append(f"- **Topic {tid}**: {', '.join(words)}")

    lines += [
        "",
        "## A second attempt: lowering min_topic_size to 5",
        "",
        "Tried to force finer granularity, comparable to LDA's 14 topics. It "
        "made the catch-all worse instead of better: 242/296 docs (82%) in "
        "one topic, down to 3 topics total (outliers dropped from 43 to 8, "
        "absorbed into the catch-all rather than better separated). This "
        "points to UMAP's default neighborhood size (n_neighbors=15), tuned "
        "for corpora orders of magnitude larger than 296 documents, rather "
        "than min_topic_size - not pursued further (see conversation log for "
        "the decision to stop here rather than tune UMAP too).",
        "",
        "## Conclusion",
        "",
        "**LDA is the topic model used going forward** (lda_topics.parquet, "
        "already exported for Phase 8's dashboard). BERTopic's default "
        "UMAP+HDBSCAN pipeline collapses most of the corpus into 1-2 broad "
        "clusters (Ukraine, Israel/Gaza) plus a majority catch-all, regardless "
        "of min_topic_size direction, on a corpus this small (296 documents). "
        "This reads as a corpus-size limitation of BERTopic's default "
        "clustering approach rather than evidence that LDA's topics are "
        "objectively better - BERTopic is designed for, and typically "
        "evaluated on, corpora several orders of magnitude larger.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "pm_contributions_clean.parquet")
    docs = build_topic_documents(df)

    model, topics = fit_bertopic(docs["text"].tolist(), min_topic_size=MIN_TOPIC_SIZE)
    info = model.get_topic_info()
    n_outliers = (
        int(info.loc[info["Topic"] == -1, "Count"].sum()) if -1 in info["Topic"].values else 0
    )
    topic_sizes = info[info["Topic"] != -1].set_index("Topic")["Count"].sort_values(ascending=False)

    bertopic_words = get_topic_words(model, topn=12)

    tokenized = [tokenize_for_topics(t) for t in docs["text"]]
    phraser = build_bigram_phraser(tokenized)
    tokenized = apply_bigrams(tokenized, phraser)
    dictionary = corpora.Dictionary(tokenized)
    cm = CoherenceModel(
        topics=list(bertopic_words.values()),
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v",
    )
    bertopic_coherence = cm.get_coherence()

    write_report(
        bertopic_words,
        bertopic_coherence,
        topic_sizes,
        n_outliers,
        len(docs),
        PROCESSED_DIR / "phase5_topic_comparison_report.md",
    )
    print(f"BERTopic coherence: {bertopic_coherence:.3f}")
    print("Wrote phase5_topic_comparison_report.md")


if __name__ == "__main__":
    main()
