from hansard_pm_nlp.lda import (
    build_dictionary_and_corpus,
    doc_topic_matrix,
    get_top_words,
    train_lda,
)


def _toy_docs():
    econ = ["economy", "tax", "growth", "inflation", "budget"] * 4
    health = ["nhs", "hospital", "doctor", "patient", "health"] * 4
    return [econ, health] * 6


def test_build_dictionary_and_corpus_shapes():
    dictionary, corpus = build_dictionary_and_corpus(_toy_docs(), no_below=1, no_above=1.0)
    assert len(corpus) == 12
    assert len(dictionary) > 0


def test_build_dictionary_and_corpus_filters_rare_words():
    docs = _toy_docs()
    docs[0] = docs[0] + ["onceonly"]
    dictionary, _ = build_dictionary_and_corpus(docs, no_below=2, no_above=1.0)
    assert "onceonly" not in dictionary.token2id


def test_train_lda_produces_requested_number_of_topics():
    dictionary, corpus = build_dictionary_and_corpus(_toy_docs(), no_below=1, no_above=1.0)
    model = train_lda(corpus, dictionary, num_topics=2, passes=5)
    assert model.num_topics == 2


def test_train_lda_is_reproducible_with_fixed_seed():
    dictionary, corpus = build_dictionary_and_corpus(_toy_docs(), no_below=1, no_above=1.0)
    model_a = train_lda(corpus, dictionary, num_topics=2, seed=42, passes=5)
    model_b = train_lda(corpus, dictionary, num_topics=2, seed=42, passes=5)
    assert model_a.get_topics().tolist() == model_b.get_topics().tolist()


def test_get_top_words_returns_one_entry_per_topic():
    dictionary, corpus = build_dictionary_and_corpus(_toy_docs(), no_below=1, no_above=1.0)
    model = train_lda(corpus, dictionary, num_topics=2, passes=5)
    top_words = get_top_words(model, topn=5)
    assert set(top_words) == {0, 1}
    assert len(top_words[0]) == 5


def test_doc_topic_matrix_has_one_row_per_document():
    dictionary, corpus = build_dictionary_and_corpus(_toy_docs(), no_below=1, no_above=1.0)
    model = train_lda(corpus, dictionary, num_topics=2, passes=5)
    matrix = doc_topic_matrix(model, corpus)
    assert len(matrix) == len(corpus)
    assert "dominant_topic" in matrix.columns


def test_doc_topic_matrix_weights_sum_to_roughly_one():
    dictionary, corpus = build_dictionary_and_corpus(_toy_docs(), no_below=1, no_above=1.0)
    model = train_lda(corpus, dictionary, num_topics=2, passes=5)
    matrix = doc_topic_matrix(model, corpus)
    topic_cols = [c for c in matrix.columns if c.startswith("topic_")]
    row_sums = matrix[topic_cols].sum(axis=1)
    assert all(abs(s - 1.0) < 0.01 for s in row_sums)
