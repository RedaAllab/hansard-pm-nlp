import math

from hansard_pm_nlp.hedging import booster_rate, hedge_rate, net_certainty


def test_hedge_rate_detects_modal_hedges():
    text = "It might possibly work but we could be wrong."
    assert hedge_rate(text) > 0


def test_hedge_rate_zero_for_assertive_text():
    text = "We will deliver this. It is guaranteed."
    assert hedge_rate(text) == 0


def test_booster_rate_detects_certainty_markers():
    text = "We will absolutely and certainly deliver this."
    assert booster_rate(text) > 0


def test_booster_rate_zero_for_hedged_text():
    text = "It might perhaps possibly work."
    assert booster_rate(text) == 0


def test_net_certainty_positive_for_assertive_text():
    text = "We will absolutely deliver this. It is guaranteed and certain."
    assert net_certainty(text) > 0


def test_net_certainty_negative_for_hedged_text():
    text = "It might possibly work, but it could perhaps fail too."
    assert net_certainty(text) < 0


def test_hedge_rate_empty_text_is_nan():
    assert math.isnan(hedge_rate(""))


def test_hedge_rate_is_case_insensitive():
    assert hedge_rate("MIGHT") == hedge_rate("might")
