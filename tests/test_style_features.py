import math

from hansard_pm_nlp.style_features import (
    build_style_features,
    function_word_rates,
    load_nlp,
    pos_tag_distribution,
)


def test_function_word_rates_counts_known_word():
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    rates = function_word_rates(tokens)
    assert rates["fw_the"] == 2 / 6
    assert rates["fw_on"] == 1 / 6


def test_function_word_rates_zero_for_absent_word():
    tokens = ["cats", "dogs", "run"]
    rates = function_word_rates(tokens)
    assert rates["fw_the"] == 0


def test_function_word_rates_empty_tokens_is_nan():
    rates = function_word_rates([])
    assert math.isnan(rates["fw_the"])


def test_pos_tag_distribution_sums_to_at_most_one():
    nlp = load_nlp()
    dist = pos_tag_distribution("The Prime Minister will address the House today.", nlp)
    assert 0 < sum(dist.values()) <= 1


def test_pos_tag_distribution_detects_verbs():
    nlp = load_nlp()
    dist = pos_tag_distribution("We will deliver this and we will win.", nlp)
    assert dist["pos_VERB"] > 0 or dist["pos_AUX"] > 0


def test_build_style_features_returns_expected_keys():
    features = build_style_features("The Government will deliver this important change.")
    assert "ttr" in features
    assert "hedge_rate" in features
    assert "fw_the" in features
    assert "pos_VERB" in features
