"""Per-PM sentiment (VADER vs. transformer) and hedging/certainty summary -
Phase 4 of PROJECT_SUMMARY.md. Descriptive only: no hypothesis testing here,
that's Phase 7's event-study layer.

Usage:
    python -m hansard_pm_nlp.affect
"""

import datetime as dt
from pathlib import Path

import pandas as pd

from hansard_pm_nlp.hedging import booster_rate, hedge_rate, net_certainty
from hansard_pm_nlp.sentiment import transformer_scores, vader_compound

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def score_corpus(df: pd.DataFrame, transformer_batch_size: int = 16) -> pd.DataFrame:
    df = df.copy()
    df["vader_compound"] = df["contribution_text"].map(vader_compound)
    df["transformer_score"] = transformer_scores(
        df["contribution_text"].tolist(), batch_size=transformer_batch_size
    )
    df["hedge_rate"] = df["contribution_text"].map(hedge_rate)
    df["booster_rate"] = df["contribution_text"].map(booster_rate)
    df["net_certainty"] = df["contribution_text"].map(net_certainty)
    df["sentiment_sign_agree"] = (df["vader_compound"] > 0) == (df["transformer_score"] > 0)
    return df


def build_pm_summary(scored: pd.DataFrame) -> pd.DataFrame:
    agg = scored.groupby("pm_name").agg(
        n_contributions=("contribution_text", "size"),
        mean_vader_compound=("vader_compound", "mean"),
        mean_transformer_score=("transformer_score", "mean"),
        sentiment_sign_agreement=("sentiment_sign_agree", "mean"),
        mean_hedge_rate=("hedge_rate", "mean"),
        mean_booster_rate=("booster_rate", "mean"),
        mean_net_certainty=("net_certainty", "mean"),
    )
    return agg.reset_index().sort_values("pm_name").reset_index(drop=True)


def build_pmqs_split(scored: pd.DataFrame) -> pd.DataFrame:
    """Hedging by debate type. PROJECT_SUMMARY.md §3 flags PMQs as genre-built for
    evasiveness - worth checking whether that shows up as a measurable hedging
    signal in this lexicon, not assuming it will (see affect_report.md: as of
    the first run, it doesn't - PMQs hedges *less* than other debates on this
    lexicon, which may mean the lexicon misses PMQs-specific evasion tactics
    like redirecting to the other party rather than hedging within the sentence).
    """
    agg = scored.groupby("is_pmqs").agg(
        n_contributions=("contribution_text", "size"),
        mean_hedge_rate=("hedge_rate", "mean"),
        mean_net_certainty=("net_certainty", "mean"),
    )
    return agg.reset_index()


def write_report(pm_summary: pd.DataFrame, pmqs_split: pd.DataFrame, path: Path) -> None:
    generated_at = dt.datetime.now(dt.UTC).isoformat()
    lines = [
        "# Phase 4 - Sentiment, hedging, certainty",
        "",
        f"Generated: {generated_at}",
        "",
        "Descriptive only. `vader_compound` in [-1, 1] (lexicon-based); "
        "`transformer_score` in [-1, 1], signed by the "
        f"{'distilbert-base-uncased-finetuned-sst-2-english'} label (contextual, binary). "
        "`sentiment_sign_agreement` is the share of contributions where both methods "
        "agree on positive vs. negative.",
        "",
        "## Per PM",
        "",
        "| PM | N | Mean VADER | Mean transformer | Sign agreement | Mean hedge rate | "
        "Mean booster rate | Mean net certainty |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in pm_summary.iterrows():
        lines.append(
            f"| {row['pm_name']} | {row['n_contributions']} | "
            f"{row['mean_vader_compound']:.3f} | {row['mean_transformer_score']:.3f} | "
            f"{row['sentiment_sign_agreement']:.1%} | {row['mean_hedge_rate']:.4f} | "
            f"{row['mean_booster_rate']:.4f} | {row['mean_net_certainty']:.4f} |"
        )

    lines += [
        "",
        "## Hedging by debate type",
        "",
        "| Is PMQs | N | Mean hedge rate | Mean net certainty |",
        "|---|---|---|---|",
    ]
    for _, row in pmqs_split.iterrows():
        lines.append(
            f"| {row['is_pmqs']} | {row['n_contributions']} | "
            f"{row['mean_hedge_rate']:.4f} | {row['mean_net_certainty']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "pm_contributions_clean.parquet")
    scored = score_corpus(df)
    scored.drop(columns=["contribution_text_raw"]).to_parquet(
        PROCESSED_DIR / "pm_contributions_affect.parquet", index=False
    )

    pm_summary = build_pm_summary(scored)
    pmqs_split = build_pmqs_split(scored)
    pm_summary.to_csv(PROCESSED_DIR / "affect_summary.csv", index=False)
    write_report(pm_summary, pmqs_split, PROCESSED_DIR / "affect_report.md")
    print("Wrote affect_summary.csv, affect_report.md, pm_contributions_affect.parquet")


if __name__ == "__main__":
    main()
