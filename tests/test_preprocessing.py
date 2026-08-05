from hansard_pm_nlp.preprocessing import (
    apply_bigrams,
    build_bigram_phraser,
    tokenize_for_topics,
)


def test_tokenize_for_topics_removes_hansard_address_terms():
    text = "The hon. Gentleman is right, if I may say to my right hon. Friend."
    tokens = tokenize_for_topics(text)
    for term in ["hon", "right", "gentleman", "friend", "may"]:
        assert term not in tokens


def test_tokenize_for_topics_keeps_topical_words():
    text = "The government announced a new energy security strategy."
    tokens = tokenize_for_topics(text)
    for term in ["government", "energy", "security", "strategy"]:
        assert term in tokens


def test_tokenize_for_topics_drops_single_char_tokens():
    text = "The government's plan is ready."
    tokens = tokenize_for_topics(text)
    assert "s" not in tokens


def test_build_bigram_phraser_merges_frequent_pairs():
    docs = [["free", "trade", "deal", "today"] for _ in range(20)]
    phraser = build_bigram_phraser(docs, min_count=1, threshold=0.1)
    merged = apply_bigrams(docs, phraser)
    assert "free_trade" in merged[0]


def test_build_bigram_phraser_does_not_merge_rare_pairs():
    docs = [["free", "trade"]] + [["completely", "unrelated", "words", "here"] for _ in range(20)]
    phraser = build_bigram_phraser(docs, min_count=5, threshold=10.0)
    merged = apply_bigrams(docs, phraser)
    assert "free_trade" not in merged[0]
