"""Lexical baseline: frequencies, n-grams, TF-IDF, readability, lexical
diversity. The descriptive floor established before any modeling (PROJECT_SUMMARY.md,
Phase 3) - no classification or inference happens here.
"""

import re
from collections import Counter

import textstat
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

_WORD_RE = re.compile(r"[a-zA-Z']+")


def tokenize_words(text: str) -> list[str]:
    """Lowercase word tokens, alphabetic + apostrophes only (drops numbers/punctuation)."""
    return _WORD_RE.findall(text.lower())


def type_token_ratio(tokens: list[str]) -> float:
    """Unique tokens / total tokens. Biased by text length - use mtld() for
    comparisons across texts of very different length (this corpus ranges
    from one-word interjections to multi-page statements).
    """
    if not tokens:
        return float("nan")
    return len(set(tokens)) / len(tokens)


def _mtld_pass(tokens: list[str], threshold: float) -> float:
    factor_count = 0
    types: set[str] = set()
    segment_len = 0
    for token in tokens:
        segment_len += 1
        types.add(token)
        if len(types) / segment_len <= threshold:
            factor_count += 1
            types = set()
            segment_len = 0
    if segment_len > 0:
        ttr = len(types) / segment_len
        factor_count += (1 - ttr) / (1 - threshold) if ttr < 1 else 0
    return len(tokens) / factor_count if factor_count > 0 else float(len(tokens))


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010): the mean
    number of tokens needed for the running TTR to drop to `threshold`,
    averaged over a forward and a backward pass. Length-independent, unlike
    type_token_ratio, which is essential here given how much contribution
    length varies (a one-word "No." vs a multi-page statement).
    """
    if len(tokens) < 2:
        return float("nan")
    forward = _mtld_pass(tokens, threshold)
    backward = _mtld_pass(list(reversed(tokens)), threshold)
    return (forward + backward) / 2


def flesch_kincaid_grade(text: str) -> float:
    return textstat.flesch_kincaid_grade(text)


def mean_words_per_sentence(text: str) -> float:
    sentences = textstat.sentence_count(text)
    words = textstat.lexicon_count(text, removepunct=True)
    return words / sentences if sentences else float("nan")


def top_ngrams(tokens: list[str], n: int = 2, top_k: int = 20) -> list[tuple[str, int]]:
    grams = zip(*(tokens[i:] for i in range(n)), strict=False)
    counts = Counter(" ".join(g) for g in grams)
    return counts.most_common(top_k)


def tfidf_top_terms_by_group(
    group_texts: dict[str, str],
    top_k: int = 20,
    extra_stopwords: frozenset[str] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """TF-IDF over one document per group (e.g. one document per PM), so a
    term's score reflects how distinctive it is to that PM relative to the
    others in `group_texts`, not its raw frequency.

    With only one document per group, IDF alone barely suppresses shared
    vocabulary that's simply used at different rates (e.g. "hon", "friend") -
    pass `extra_stopwords` (e.g. preprocessing.STOPWORDS) to remove Hansard's
    parliamentary-address vocabulary the same way the LDA pipeline already
    does, on top of the standard English stopword list.
    """
    names = list(group_texts)
    stop_words = ENGLISH_STOP_WORDS | (extra_stopwords or frozenset())
    vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=5000)
    matrix = vectorizer.fit_transform([group_texts[name] for name in names])
    terms = vectorizer.get_feature_names_out()

    result = {}
    for i, name in enumerate(names):
        row = matrix[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_k]
        result[name] = [(terms[j], float(row[j])) for j in top_idx if row[j] > 0]
    return result
