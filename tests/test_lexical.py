import math

from hansard_pm_nlp.lexical import (
    flesch_kincaid_grade,
    mean_words_per_sentence,
    mtld,
    tfidf_top_terms_by_group,
    tokenize_words,
    top_ngrams,
    type_token_ratio,
)


def test_tokenize_words_lowercases_and_strips_punctuation():
    assert tokenize_words("Hello, World! It's fine.") == ["hello", "world", "it's", "fine"]


def test_tokenize_words_drops_numbers():
    assert tokenize_words("We have 20,000 police.") == ["we", "have", "police"]


def test_type_token_ratio_all_unique_is_one():
    assert type_token_ratio(["a", "b", "c"]) == 1.0


def test_type_token_ratio_all_repeated():
    assert type_token_ratio(["a", "a", "a", "a"]) == 0.25


def test_type_token_ratio_empty_is_nan():
    assert math.isnan(type_token_ratio([]))


def test_mtld_higher_for_more_diverse_text():
    repetitive = ["the", "cat", "sat"] * 50
    diverse = [f"word{i}" for i in range(150)]
    assert mtld(diverse) > mtld(repetitive)


def test_mtld_short_input_is_nan():
    assert math.isnan(mtld(["only", "one", "pair"][:1]))


def test_mtld_positive_for_reasonable_text():
    tokens = tokenize_words(
        "The quick brown fox jumps over the lazy dog. "
        "A completely different sentence follows with new words entirely."
    )
    assert mtld(tokens) > 0


def test_flesch_kincaid_grade_higher_for_more_complex_text():
    simple = "The cat sat. The dog ran. I see a cat."
    complex_ = (
        "The precipitous deterioration of macroeconomic indicators "
        "necessitates an immediate and comprehensive policy recalibration."
    )
    assert flesch_kincaid_grade(complex_) > flesch_kincaid_grade(simple)


def test_mean_words_per_sentence():
    text = "One two three. Four five six."
    assert mean_words_per_sentence(text) == 3.0


def test_top_ngrams_bigrams_counts_correctly():
    tokens = tokenize_words("the cat sat on the mat the cat ran")
    result = dict(top_ngrams(tokens, n=2, top_k=10))
    assert result["the cat"] == 2


def test_top_ngrams_respects_top_k():
    tokens = tokenize_words("a b c d e f g h")
    assert len(top_ngrams(tokens, n=1, top_k=3)) == 3


def test_tfidf_top_terms_by_group_surfaces_distinctive_terms():
    group_texts = {
        "pm_a": "brexit brexit brexit sovereignty sovereignty parliament",
        "pm_b": "covid covid covid lockdown lockdown parliament",
    }
    result = tfidf_top_terms_by_group(group_texts, top_k=3)
    scores_a = dict(result["pm_a"])
    scores_b = dict(result["pm_b"])
    assert scores_a["brexit"] > scores_a["parliament"]
    assert scores_b["covid"] > scores_b["parliament"]


def test_tfidf_top_terms_by_group_drops_extra_stopwords():
    # "hon" appears in both groups at a differential rate - with only one
    # document per group, IDF alone doesn't suppress it (see lexical.py
    # docstring); extra_stopwords must filter it out explicitly.
    group_texts = {
        "pm_a": "hon hon hon hon brexit sovereignty parliament",
        "pm_b": "hon hon covid lockdown parliament",
    }
    result = tfidf_top_terms_by_group(
        group_texts, top_k=5, extra_stopwords=frozenset({"hon"})
    )
    terms_a = {term for term, _ in result["pm_a"]}
    terms_b = {term for term, _ in result["pm_b"]}
    assert "hon" not in terms_a
    assert "hon" not in terms_b
