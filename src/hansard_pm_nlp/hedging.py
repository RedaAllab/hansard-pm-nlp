"""Custom hedging/certainty lexicon: modals, hedge verbs, and epistemic
adverbs (uncertainty markers) vs. booster adverbs/modals (certainty markers).
Especially relevant to PMQs, a genre built around evasiveness (PROJECT_SUMMARY.md).

Single-token markers only - multi-word hedges ("sort of", "kind of") are a
known gap, acceptable for a first lexicon; extend HEDGE_WORDS/BOOSTER_WORDS
rather than adding phrase matching unless the single-token version proves
too coarse.
"""

from hansard_pm_nlp.lexical import tokenize_words

# Modal verbs and epistemic markers expressing possibility, not certainty.
HEDGE_WORDS = frozenset(
    {
        "might",
        "may",
        "could",
        "would",
        "should",
        "perhaps",
        "possibly",
        "probably",
        "likely",
        "presumably",
        "apparently",
        "seem",
        "seems",
        "seemed",
        "appear",
        "appears",
        "appeared",
        "suggest",
        "suggests",
        "suggested",
        "believe",
        "believes",
        "think",
        "thinks",
        "assume",
        "assumes",
        "somewhat",
        "roughly",
        "approximately",
    }
)

# Modal verbs and adverbs expressing certainty/emphasis - the counterpart to
# HEDGE_WORDS, so a text's net stance isn't judged on hedges alone.
BOOSTER_WORDS = frozenset(
    {
        "will",
        "must",
        "definitely",
        "certainly",
        "absolutely",
        "undoubtedly",
        "clearly",
        "obviously",
        "always",
        "never",
        "guarantee",
        "guaranteed",
        "guarantees",
        "surely",
    }
)


def _marker_rate(text: str, markers: frozenset[str]) -> float:
    tokens = tokenize_words(text)
    if not tokens:
        return float("nan")
    return sum(1 for t in tokens if t in markers) / len(tokens)


def hedge_rate(text: str) -> float:
    return _marker_rate(text, HEDGE_WORDS)


def booster_rate(text: str) -> float:
    return _marker_rate(text, BOOSTER_WORDS)


def net_certainty(text: str) -> float:
    """booster_rate - hedge_rate: positive leans assertive, negative leans hedged."""
    return booster_rate(text) - hedge_rate(text)
