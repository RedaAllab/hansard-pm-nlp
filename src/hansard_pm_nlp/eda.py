"""Per-PM lexical baseline: the descriptive floor established before any
modeling (PROJECT_SUMMARY.md, Phase 3).

Readability and sentence length are computed per contribution then averaged,
so one very long statement can't dominate the PM's score and sentence
boundaries are never guessed across concatenated, unrelated contributions.
TTR/MTLD and n-grams instead need a large token window to be meaningful, so
those are computed on each PM's full concatenated corpus.

Usage:
    python -m hansard_pm_nlp.eda
"""

import datetime as dt
from pathlib import Path

import pandas as pd

from hansard_pm_nlp.lexical import (
    flesch_kincaid_grade,
    mean_words_per_sentence,
    mtld,
    tfidf_top_terms_by_group,
    tokenize_words,
    top_ngrams,
    type_token_ratio,
)
from hansard_pm_nlp.preprocessing import STOPWORDS as DOMAIN_STOPWORDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _per_contribution_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["word_count"] = df["contribution_text"].map(lambda t: len(tokenize_words(t)))
    df["flesch_kincaid_grade"] = df["contribution_text"].map(flesch_kincaid_grade)
    df["mean_words_per_sentence"] = df["contribution_text"].map(mean_words_per_sentence)
    return df


def build_summary(df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    scored = _per_contribution_stats(df)

    per_pm_tokens = {
        pm: tokenize_words(" ".join(group["contribution_text"]))
        for pm, group in df.groupby("pm_name")
    }
    per_pm_text = {pm: " ".join(group["contribution_text"]) for pm, group in df.groupby("pm_name")}
    tfidf_terms = tfidf_top_terms_by_group(
        per_pm_text, top_k=top_k, extra_stopwords=DOMAIN_STOPWORDS
    )

    rows = []
    agg = scored.groupby("pm_name").agg(
        n_contributions=("contribution_text", "size"),
        total_words=("word_count", "sum"),
        mean_flesch_kincaid_grade=("flesch_kincaid_grade", "mean"),
        mean_words_per_sentence=("mean_words_per_sentence", "mean"),
        pmqs_share=("is_pmqs", "mean"),
    )
    for pm, row in agg.iterrows():
        tokens = per_pm_tokens[pm]
        top_bigrams = top_ngrams(tokens, n=2, top_k=top_k)
        rows.append(
            {
                "pm_name": pm,
                "n_contributions": int(row["n_contributions"]),
                "total_words": int(row["total_words"]),
                "type_token_ratio": type_token_ratio(tokens),
                "mtld": mtld(tokens),
                "mean_flesch_kincaid_grade": row["mean_flesch_kincaid_grade"],
                "mean_words_per_sentence": row["mean_words_per_sentence"],
                "pmqs_share": row["pmqs_share"],
                "top_bigrams": "; ".join(f"{g} ({c})" for g, c in top_bigrams),
                "top_tfidf_terms": "; ".join(f"{t} ({s:.3f})" for t, s in tfidf_terms.get(pm, [])),
            }
        )
    return pd.DataFrame(rows).sort_values("pm_name").reset_index(drop=True)


def build_mtld_over_time(df: pd.DataFrame, freq: str = "MS", min_words: int = 1500) -> pd.DataFrame:
    """Lexical diversity (MTLD) per PM, binned by `freq` (default monthly).

    build_summary() computes MTLD once per PM on their full concatenated
    corpus; this recomputes it per time bin instead, to show drift over a
    tenure. That reintroduces MTLD's own length-sensitivity at a smaller
    scale - the same problem TTR has more severely (lexical.py, radar
    caption) - so bins under `min_words` are dropped rather than plotted as
    a misleadingly precise point. Mainly affects Liz Truss (49-day tenure,
    8,842 words total): expect few or no surviving bins for her.
    """
    df = df.copy()
    df["sitting_date"] = pd.to_datetime(df["sitting_date"])

    rows = []
    for (pm, period), group in df.groupby(["pm_name", pd.Grouper(key="sitting_date", freq=freq)]):
        tokens = tokenize_words(" ".join(group["contribution_text"]))
        if len(tokens) < min_words:
            continue
        rows.append(
            {
                "pm_name": pm,
                "period": period,
                "word_count": len(tokens),
                "mtld": mtld(tokens),
            }
        )
    columns = ["pm_name", "period", "word_count", "mtld"]
    return pd.DataFrame(rows, columns=columns).sort_values(["pm_name", "period"]).reset_index(
        drop=True
    )


def write_report(summary: pd.DataFrame, path: Path) -> None:
    generated_at = dt.datetime.now(dt.UTC).isoformat()
    lines = [
        "# Phase 3 - Lexical baseline",
        "",
        f"Generated: {generated_at}",
        "",
        "Descriptive statistics only - no hypothesis tests, no modeling. "
        "Readability/sentence length are averaged per contribution; lexical "
        "diversity (TTR, MTLD) and n-grams are computed on each PM's full "
        "concatenated corpus (see hansard_pm_nlp.eda docstring for why).",
        "",
        "`mtld_over_time.parquet` additionally bins MTLD by month per PM "
        "(build_mtld_over_time()), dropping any bin under 1,500 words so a "
        "sparse month doesn't get plotted as a misleadingly precise point - "
        "see the dashboard's Overview tab.",
        "",
    ]
    for _, row in summary.iterrows():
        lines += [
            f"## {row['pm_name']}",
            "",
            f"- Contributions: {row['n_contributions']} ({row['total_words']} words, "
            f"{row['pmqs_share']:.0%} PMQs)",
            f"- Type-token ratio: {row['type_token_ratio']:.4f} "
            f"(biased by corpus size - see MTLD instead for cross-PM comparison)",
            f"- MTLD: {row['mtld']:.1f}",
            f"- Mean Flesch-Kincaid grade: {row['mean_flesch_kincaid_grade']:.2f}",
            f"- Mean words per sentence: {row['mean_words_per_sentence']:.1f}",
            f"- Top bigrams: {row['top_bigrams']}",
            f"- Top TF-IDF terms (distinctive vs. other PMs): {row['top_tfidf_terms']}",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "pm_contributions_clean.parquet")
    summary = build_summary(df)
    summary.to_csv(PROCESSED_DIR / "eda_summary.csv", index=False)
    write_report(summary, PROCESSED_DIR / "eda_report.md")

    mtld_over_time = build_mtld_over_time(df)
    mtld_over_time.to_parquet(PROCESSED_DIR / "mtld_over_time.parquet", index=False)

    print(
        f"Wrote {PROCESSED_DIR / 'eda_summary.csv'}, eda_report.md, "
        "and mtld_over_time.parquet"
    )


if __name__ == "__main__":
    main()
