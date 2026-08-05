"""Topic-modeling preprocessing: stopwords, tokenization, bigrams.

Phase 3's TF-IDF (eda_report.md) was dominated by parliamentary address
vocabulary - "hon", "right", "friend", "gentleman", "house" - rather than
policy content, because that's how MPs address each other in Hansard, not
because those words carry topic signal. HANSARD_STOPWORDS removes that
address vocabulary on top of the standard English stopword list; it
deliberately does NOT remove words like "government", "security", or
"labour" that are ambiguous between procedural and topical use - see the
module-level comment below for the reasoning.
"""

from gensim.models.phrases import Phraser, Phrases
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from hansard_pm_nlp.lexical import tokenize_words

# Terms for addressing/referring to other MPs and chamber roles in Hansard's
# formal register (e.g. "my right hon. Friend", "the hon. Gentleman", "Mr
# Speaker"). Excludes words that are ambiguous between procedural and topical
# use (e.g. "government", "secretary", "labour", "security") - those are left
# for the topic model to sort out rather than removed on a guess.
HANSARD_STOPWORDS = frozenset(
    {
        "hon",
        "right",
        "friend",
        "gentleman",
        "lady",
        "house",
        "speaker",
        "mr",
        "member",
        "members",
        "chamber",
        "colleague",
        "colleagues",
        "chair",
        "deputy",
    }
)

# Politeness/filler vocabulary observed polluting the first K=15 LDA run
# (lda_top_words_k15.txt): thank/raising merged into a meaningless
# "thank_raising" bigram, and words like "fantastic"/"grateful"/"certainly"
# showed up as top words in multiple otherwise-unrelated topics - PMQs-style
# politeness formulas, not topic content. Same "left for the model" caveat as
# HANSARD_STOPWORDS: only removed here because they showed up as noise in
# practice, not guessed in advance.
FILLER_STOPWORDS = frozenset(
    {
        "thank",
        "thanks",
        "raising",
        "really",
        "important",
        "grateful",
        "certainly",
        "fantastic",
        "pleased",
        "welcome",
        "congratulate",
        "weekend",
        "completely",
    }
)

STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | HANSARD_STOPWORDS | FILLER_STOPWORDS


def tokenize_for_topics(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed, single-character tokens
    dropped (leftover from apostrophe splitting, e.g. "s" from "government's").
    """
    return [t for t in tokenize_words(text) if t not in STOPWORDS and len(t) > 1]


def build_bigram_phraser(
    tokenized_docs: list[list[str]], min_count: int = 5, threshold: float = 10.0
) -> Phraser:
    """Fit a bigram detector (e.g. "free" + "trade" -> "free_trade") on the
    full corpus of tokenized documents.
    """
    phrases = Phrases(tokenized_docs, min_count=min_count, threshold=threshold)
    return Phraser(phrases)


def apply_bigrams(tokenized_docs: list[list[str]], phraser: Phraser) -> list[list[str]]:
    return [phraser[doc] for doc in tokenized_docs]
