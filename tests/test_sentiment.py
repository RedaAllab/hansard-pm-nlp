from hansard_pm_nlp.sentiment import (
    transformer_scores,
    transformer_signed_score,
    vader_compound,
)


def test_vader_compound_positive():
    assert vader_compound("This is wonderful, excellent news!") > 0.5


def test_vader_compound_negative():
    assert vader_compound("This is terrible, awful and disastrous.") < -0.5


def test_vader_compound_neutral_near_zero():
    assert abs(vader_compound("The meeting is at three o'clock.")) < 0.2


def test_transformer_signed_score_positive_label():
    assert transformer_signed_score({"label": "POSITIVE", "score": 0.9}) == 0.9


def test_transformer_signed_score_negative_label():
    assert transformer_signed_score({"label": "NEGATIVE", "score": 0.8}) == -0.8


def test_transformer_scores_matches_expected_sign():
    scores = transformer_scores(
        ["This is a wonderful and fantastic achievement.", "This is a disastrous failure."]
    )
    assert scores[0] > 0
    assert scores[1] < 0
