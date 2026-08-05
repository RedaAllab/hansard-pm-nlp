# Phase 5 - LDA vs BERTopic comparison

Generated: 2026-08-05T21:08:27.883119+00:00

Same 296 documents (PM x sitting date), same STOPWORDS list applied to keyword extraction in both methods (c-TF-IDF for BERTopic, dictionary filtering for LDA) - see phase5_lda_report.md for the LDA side.

## Coherence (c_v, comparable metric across both)

| Method | Topics | c_v coherence |
|---|---|---|
| LDA (K=14) | 14 | 0.554 |
| BERTopic (min_topic_size=10) | 4 | 0.468 |

Coherence alone understates the gap: BERTopic's number (0.468) isn't far below LDA's (0.554), but that number doesn't capture that 180/296 documents (61%) land in a single catch-all topic - see below.

## BERTopic topic sizes

| Topic | N docs |
|---|---|
| 0 | 180 |
| 1 | 32 |
| 2 | 21 |
| 3 | 20 |
| -1 (outliers) | 43 |

## BERTopic topics

- **Topic 0**: people, country, government, support, know, think, work, new, labour, just
- **Topic 1**: people, country, think, support, way, nhs, virus, government, care, want
- **Topic 2**: ukraine, nato, putin, russia, support, russian, people, country, uk, president
- **Topic 3**: region, israel, aid, support, people, gaza, security, ensure, continue, hamas

## A second attempt: lowering min_topic_size to 5

Tried to force finer granularity, comparable to LDA's 14 topics. It made the catch-all worse instead of better: 242/296 docs (82%) in one topic, down to 3 topics total (outliers dropped from 43 to 8, absorbed into the catch-all rather than better separated). This points to UMAP's default neighborhood size (n_neighbors=15), tuned for corpora orders of magnitude larger than 296 documents, rather than min_topic_size - not pursued further (see conversation log for the decision to stop here rather than tune UMAP too).

## Conclusion

**LDA is the topic model used going forward** (lda_topics.parquet, already exported for Phase 8's dashboard). BERTopic's default UMAP+HDBSCAN pipeline collapses most of the corpus into 1-2 broad clusters (Ukraine, Israel/Gaza) plus a majority catch-all, regardless of min_topic_size direction, on a corpus this small (296 documents). This reads as a corpus-size limitation of BERTopic's default clustering approach rather than evidence that LDA's topics are objectively better - BERTopic is designed for, and typically evaluated on, corpora several orders of magnitude larger.
