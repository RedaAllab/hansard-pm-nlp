"""BERTopic: the contextual counterpart to LDA (lda.py), compared explicitly
per PROJECT_SUMMARY.md's Planned NLP analyses table.

Unlike LDA, which needs stopword-filtered bag-of-words tokens as input,
BERTopic embeds natural text (its sentence-transformer benefits from real
syntax) and only applies stopword filtering downstream, in the c-TF-IDF step
that extracts each topic's representative words. Reusing preprocessing.py's
STOPWORDS there keeps the topic *vocabulary* comparable between the two
methods even though the *input* to each model differs by necessity.

UMAP's random_state is fixed for reproducibility (CLAUDE.md §7); this also
disables UMAP's parallelism, a known and accepted tradeoff.
"""

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from hansard_pm_nlp.preprocessing import STOPWORDS

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RANDOM_SEED = 42


def build_bertopic_model(min_topic_size: int = 10) -> BERTopic:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    umap_model = UMAP(random_state=RANDOM_SEED)
    vectorizer_model = CountVectorizer(stop_words=list(STOPWORDS), ngram_range=(1, 2))
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
    )


def fit_bertopic(documents: list[str], min_topic_size: int = 10) -> tuple[BERTopic, list[int]]:
    model = build_bertopic_model(min_topic_size=min_topic_size)
    topics, _ = model.fit_transform(documents)
    return model, topics


def get_topic_words(model: BERTopic, topn: int = 12) -> dict[int, list[str]]:
    """Top words per topic, excluding the -1 outlier bucket (not a real topic)."""
    info = model.get_topic_info()
    result = {}
    for topic_id in info["Topic"]:
        if topic_id == -1:
            continue
        result[topic_id] = [w for w, _ in model.get_topic(topic_id)[:topn]]
    return result
