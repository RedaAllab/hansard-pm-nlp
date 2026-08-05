from hansard_pm_nlp.bertopic_model import fit_bertopic, get_topic_words


def _toy_docs():
    econ = " ".join(["economy", "tax", "growth", "inflation", "budget"] * 20)
    health = " ".join(["nhs", "hospital", "doctor", "patient", "health"] * 20)
    return [econ, health] * 4


def test_fit_bertopic_returns_a_topic_per_document():
    _, topics = fit_bertopic(_toy_docs(), min_topic_size=2)
    assert len(topics) == len(_toy_docs())


def test_get_topic_words_excludes_outlier_bucket():
    model, _ = fit_bertopic(_toy_docs(), min_topic_size=2)
    words = get_topic_words(model)
    assert -1 not in words
