"""Stylometric feature extraction for the PM-attribution classifier (Phase 6,
H1). Combines the descriptive layers already built in earlier phases
(lexical diversity, readability, hedging/certainty) with two classic
authorship-attribution signals not yet used in this project: individual
function-word frequencies and POS-tag distribution, both computed via spaCy.

Function words (articles, prepositions, pronouns, conjunctions, auxiliaries)
are the workhorse feature of authorship attribution since Mosteller & Wallace
(1963, the Federalist Papers) - their usage rate is largely topic-independent,
which is exactly the property H1 needs (PMs discuss different subjects, but a
genuine stylistic signature should hold regardless of subject).
"""

from collections import Counter
from functools import lru_cache

import spacy

from hansard_pm_nlp.hedging import booster_rate, hedge_rate, net_certainty
from hansard_pm_nlp.lexical import (
    flesch_kincaid_grade,
    mean_words_per_sentence,
    mtld,
    tokenize_words,
    type_token_ratio,
)

# Classic closed-class function words: determiners, prepositions, pronouns,
# conjunctions, auxiliaries. Frequency-independent of topic, unlike content words.
FUNCTION_WORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "in", "for", "on", "with", "as", "at", "by",
        "from", "this", "that", "these", "those", "which", "who", "what", "or",
        "and", "but", "if", "so", "than", "because", "although", "while",
        "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
        "will", "would", "can", "could", "shall", "should", "may", "might", "must",
        "not", "he", "she", "it", "they", "we", "you", "i", "his", "her", "their",
        "our", "your", "my", "its",
    }
)

POS_TAGS = (
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "AUX",
    "CCONJ", "SCONJ", "PART", "NUM", "PROPN", "INTJ",
)


@lru_cache(maxsize=1)
def load_nlp():
    """spaCy pipeline with only tok2vec/tagger/attribute_ruler enabled - POS
    tags are all this module needs, and the parser/ner/lemmatizer are the
    slow components on documents this long (median ~2,700 words).
    """
    return spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])


def function_word_rates(tokens: list[str]) -> dict[str, float]:
    """Rate of each function word, out of total tokens. NaN (not zero) on an
    empty document, consistent with lexical.py/hedging.py's convention.
    """
    if not tokens:
        return {f"fw_{w}": float("nan") for w in FUNCTION_WORDS}
    counts = Counter(tokens)
    n = len(tokens)
    return {f"fw_{w}": counts.get(w, 0) / n for w in FUNCTION_WORDS}


def pos_tag_distribution(text: str, nlp=None) -> dict[str, float]:
    """Proportion of tokens in each of POS_TAGS. Tags outside this set (e.g.
    PUNCT, SPACE, SYM, X) are counted in the denominator but have no column,
    so the returned rates need not sum to 1.
    """
    nlp = nlp or load_nlp()
    doc = nlp(text)
    n = len(doc)
    if n == 0:
        return {f"pos_{tag}": float("nan") for tag in POS_TAGS}
    counts = Counter(tok.pos_ for tok in doc)
    return {f"pos_{tag}": counts.get(tag, 0) / n for tag in POS_TAGS}


def build_style_features(text: str, nlp=None) -> dict[str, float]:
    """One flat dict of stylometric features for a single document: lexical
    diversity, readability, hedging/certainty, function-word rates, and POS
    distribution.
    """
    tokens = tokenize_words(text)
    features = {
        "ttr": type_token_ratio(tokens),
        "mtld": mtld(tokens),
        "flesch_kincaid": flesch_kincaid_grade(text),
        "mean_words_per_sentence": mean_words_per_sentence(text),
        "hedge_rate": hedge_rate(text),
        "booster_rate": booster_rate(text),
        "net_certainty": net_certainty(text),
    }
    features.update(function_word_rates(tokens))
    features.update(pos_tag_distribution(text, nlp))
    return features
