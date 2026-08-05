"""Sentiment: VADER (lexicon-based baseline) vs. a pretrained transformer
(contextual). Compared explicitly rather than picking one, per PROJECT_SUMMARY.md's
Planned NLP analyses table - the gap between them is itself a finding.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

# distilbert-base-uncased-finetuned-sst-2-english: binary pos/neg, no neutral
# class. Chosen as the lightweight contextual counterpart to VADER - a 3-class
# model would let the two disagree less trivially, but binary-vs-lexicon is
# still the intended comparison here (does context beat a word list at all).
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def vader_compound(text: str) -> float:
    """VADER's compound score, in [-1, 1]."""
    return _vader.polarity_scores(text)["compound"]


def build_transformer_pipeline():
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis", model=TRANSFORMER_MODEL_NAME, truncation=True, max_length=512
    )


def transformer_signed_score(result: dict) -> float:
    """Convert a pipeline result ({'label': 'POSITIVE'|'NEGATIVE', 'score': p})
    into a single signed score in [-1, 1], comparable in sign/scale to VADER's compound.
    """
    sign = 1 if result["label"] == "POSITIVE" else -1
    return sign * result["score"]


def transformer_scores(texts: list[str], batch_size: int = 16) -> list[float]:
    clf = build_transformer_pipeline()
    results = clf(texts, batch_size=batch_size)
    return [transformer_signed_score(r) for r in results]
