"""Train the final LDA model (K=14, seed=42) and export the doc-topic matrix,
top words per topic, and the methodology report - including the persistent
Ukraine/security topic duplication, documented rather than hidden.

K=14 was chosen by hand after: a coherence sweep over K=5..30 (noisy, no
clean plateau, best at K=15/0.452), then adding FILLER_STOPWORDS after
K=15's top words showed politeness-formula noise (coherence -> 0.503), then
refining K in {10,12,13,14,15} on the cleaned vocabulary (best at K=14/0.554).
See phase5_lda_report.md for the full log.

Usage:
    python -m hansard_pm_nlp.build_lda_topics
"""

import datetime as dt
from pathlib import Path

import pandas as pd

from hansard_pm_nlp.lda import (
    build_dictionary_and_corpus,
    coherence_c_v,
    doc_topic_matrix,
    get_top_words,
    train_lda,
)
from hansard_pm_nlp.preprocessing import apply_bigrams, build_bigram_phraser, tokenize_for_topics
from hansard_pm_nlp.topic_corpus import build_topic_documents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FINAL_K = 14

K_SEARCH_LOG = [
    ("Initial sweep, K=5..30, before filler stopword cleaning", 15, 0.452),
    ("Refinement after adding FILLER_STOPWORDS, K=15", 15, 0.503),
    ("Refinement after adding FILLER_STOPWORDS, K=10", 10, 0.400),
    ("Refinement after adding FILLER_STOPWORDS, K=12", 12, 0.500),
    ("Refinement after adding FILLER_STOPWORDS, K=13", 13, 0.448),
    ("Refinement after adding FILLER_STOPWORDS, K=14 (final)", 14, 0.554),
]


def write_report(top_words: dict[int, list[str]], coherence: float, path: Path) -> None:
    generated_at = dt.datetime.now(dt.UTC).isoformat()
    lines = [
        "# Phase 5 - LDA topic model",
        "",
        f"Generated: {generated_at}",
        f"Final model: K={FINAL_K}, seed=42, c_v coherence={coherence:.3f}",
        "",
        "## Document unit",
        "",
        "One document per (PM, sitting date): 296 documents after dropping 2 "
        "under 50 words. Raw contributions range from one-word interjections "
        "to multi-page statements, too uneven for stable per-document topic "
        "distributions.",
        "",
        "## Preprocessing",
        "",
        "Standard English stopwords + HANSARD_STOPWORDS (parliamentary address "
        "vocabulary: hon, right, friend, gentleman, house...) + FILLER_STOPWORDS "
        "(politeness formulas found polluting the first LDA run: thank, raising, "
        "really, important, grateful, certainly...). Bigrams via gensim Phrases. "
        "Dictionary filtered to words in >=3 documents and <=50% of documents.",
        "",
        "## K selection log",
        "",
        "No clean coherence plateau across K=5..30 (see lda_coherence_sweep.csv). "
        "K was picked by hand after inspecting topic word lists, not by "
        "maximizing coherence alone:",
        "",
        "| Step | K | c_v coherence |",
        "|---|---|---|",
    ]
    for step, k, score in K_SEARCH_LOG:
        marker = " **(final)**" if k == FINAL_K and score == coherence else ""
        lines.append(f"| {step} | {k} | {score:.3f}{marker} |")

    lines += [
        "",
        "## Known limitation: Ukraine/security topic duplication",
        "",
        "Topics 0 and 1 are both Ukraine/Russia/NATO/security, with substantial "
        "vocabulary overlap, at every K tested (10, 12, 13, 14, 15) - this did "
        "not resolve by lowering K, and lowering K instead merged genuinely "
        "unrelated topics together (e.g. Northern Ireland with Israel/Gaza at "
        "K=10). Read as a real structural feature of the corpus: PMs discuss "
        "Russia/Ukraine across distinct sub-periods (the 2022 invasion, ongoing "
        "military aid, NATO summits) with different vocabulary each time, not "
        "as a preprocessing or K artifact to fix.",
        "",
        "## Topics (K=14)",
        "",
    ]
    for i, words in top_words.items():
        lines.append(f"- **Topic {i}**: {', '.join(words)}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "pm_contributions_clean.parquet")
    docs = build_topic_documents(df)

    tokenized = [tokenize_for_topics(t) for t in docs["text"]]
    phraser = build_bigram_phraser(tokenized)
    tokenized = apply_bigrams(tokenized, phraser)

    dictionary, corpus = build_dictionary_and_corpus(tokenized)
    model = train_lda(corpus, dictionary, num_topics=FINAL_K)
    coherence = coherence_c_v(model, tokenized, dictionary)

    matrix = doc_topic_matrix(model, corpus)
    result = pd.concat(
        [docs[["pm_name", "sitting_date", "n_contributions", "pmqs_share"]], matrix], axis=1
    )
    result.to_parquet(PROCESSED_DIR / "lda_topics.parquet", index=False)

    top_words = get_top_words(model, topn=12)
    write_report(top_words, coherence, PROCESSED_DIR / "phase5_lda_report.md")

    model.save(str(PROCESSED_DIR / "lda_model.gensim"))
    dictionary.save(str(PROCESSED_DIR / "lda_dictionary.gensim"))

    print(f"Coherence: {coherence:.3f}")
    print("Wrote lda_topics.parquet, phase5_lda_report.md, lda_model.gensim")


if __name__ == "__main__":
    main()
