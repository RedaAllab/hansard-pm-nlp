"""LDA topic model: dictionary/corpus construction, training, and a coherence
(c_v) sweep over K. The sweep's result is reviewed by hand (see
run_coherence_sweep.py) before a final K is chosen - not auto-selected -
per CLAUDE.md §9's rule that this phase is discussed, not silently generated.

Random seed fixed for reproducibility (CLAUDE.md §7).
"""

import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel

RANDOM_SEED = 42


def build_dictionary_and_corpus(
    tokenized_docs: list[list[str]], no_below: int = 3, no_above: float = 0.5
) -> tuple[corpora.Dictionary, list[list[tuple[int, int]]]]:
    """no_below=3: a word must appear in at least 3 documents (298 total) to
    survive - single-sitting jargon isn't a corpus-level topic signal.
    no_above=0.5: drop words in over half the documents - too common to
    distinguish topics, the gensim equivalent of the STOPWORDS list for
    corpus-specific high-frequency terms it wouldn't otherwise catch.
    """
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    return dictionary, corpus


def train_lda(
    corpus: list[list[tuple[int, int]]],
    dictionary: corpora.Dictionary,
    num_topics: int,
    seed: int = RANDOM_SEED,
    passes: int = 10,
) -> LdaModel:
    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=seed,
        passes=passes,
        alpha="auto",
        eta="auto",
    )


def coherence_c_v(
    model: LdaModel, tokenized_docs: list[list[str]], dictionary: corpora.Dictionary
) -> float:
    cm = CoherenceModel(model=model, texts=tokenized_docs, dictionary=dictionary, coherence="c_v")
    return cm.get_coherence()


def coherence_sweep(
    tokenized_docs: list[list[str]], k_grid: list[int], seed: int = RANDOM_SEED
) -> pd.DataFrame:
    dictionary, corpus = build_dictionary_and_corpus(tokenized_docs)
    rows = []
    for k in k_grid:
        model = train_lda(corpus, dictionary, k, seed=seed)
        score = coherence_c_v(model, tokenized_docs, dictionary)
        rows.append({"k": k, "coherence_c_v": score})
    return pd.DataFrame(rows)


def get_top_words(model: LdaModel, topn: int = 12) -> dict[int, list[str]]:
    return {i: [w for w, _ in model.show_topic(i, topn=topn)] for i in range(model.num_topics)}


def doc_topic_matrix(model: LdaModel, corpus: list[list[tuple[int, int]]]) -> pd.DataFrame:
    """One row per document, one column per topic (weight), plus `dominant_topic`.
    Missing topics in gensim's sparse per-doc output are filled with 0.
    """
    rows = []
    for bow in corpus:
        weights = dict(model.get_document_topics(bow, minimum_probability=0.0))
        rows.append([weights.get(t, 0.0) for t in range(model.num_topics)])
    matrix = pd.DataFrame(rows, columns=[f"topic_{i}" for i in range(model.num_topics)])
    matrix["dominant_topic"] = matrix.values.argmax(axis=1)
    return matrix
