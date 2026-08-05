"""Apply cleaning and debate-type tagging to the raw corpus, export the
processed parquet with its own data_README.md and schema.json.

Usage:
    python -m hansard_pm_nlp.build_corpus
"""

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from hansard_pm_nlp.cleaning import clean_contribution_text, clean_debate_section
from hansard_pm_nlp.tagging import add_is_pmqs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "input"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_processed_contributions(
    input_path: Path = INPUT_DIR / "pm_contributions.parquet",
) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df["contribution_text_raw"] = df["contribution_text"]
    df["contribution_text"] = df["contribution_text_raw"].map(clean_contribution_text)
    df["debate_section"] = df["debate_section"].map(clean_debate_section)
    df = add_is_pmqs(df)
    return df


def write_schema_json(df: pd.DataFrame, path: Path) -> None:
    schema = {
        "table": "pm_contributions_clean",
        "columns": {col: str(df[col].dtype) for col in df.columns},
        "row_count": len(df),
    }
    path.write_text(json.dumps(schema, indent=2))


def write_data_readme(df: pd.DataFrame, path: Path) -> None:
    generated_at = dt.datetime.now(dt.UTC).isoformat()
    debate_counts = df["debate_section"].value_counts()
    lines = [
        "# Data README - processed corpus",
        "",
        f"Generated: {generated_at}",
        f"Row count: {len(df)}",
        "",
        "## Cleaning applied",
        "",
        "- `contribution_text_raw`: original text, untouched, kept for audit.",
        "- `contribution_text`: HTML tags/entities stripped, whitespace normalized "
        "(see hansard_pm_nlp.cleaning.clean_contribution_text).",
        "- `debate_section`: leading/trailing whitespace stripped, merging variants "
        "that were previously distinct categories (e.g. ' Covid-19 Update' vs 'Covid-19 Update').",
        "- `is_pmqs`: True where `debate_section == 'Engagements'` (Hansard's own filing "
        "for Prime Minister's Questions).",
        "- No rows dropped: the 5 exact-duplicate-text rows found (same PM/date/text) are "
        "distinct short interjections ('Sit down.', 'No.') with distinct contribution_ext_id "
        "and debate order, not ingestion artifacts. See Phase 2 notes in PROJECT_SUMMARY.md.",
        "",
        f"PMQs contributions: {int(df['is_pmqs'].sum())} / {len(df)}",
        "",
        "## debate_section value counts (top 15)",
        "",
    ]
    for name, count in debate_counts.head(15).items():
        lines.append(f"- {name}: {count}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_processed_contributions()
    df.to_parquet(PROCESSED_DIR / "pm_contributions_clean.parquet", index=False)
    write_schema_json(df, PROCESSED_DIR / "schema.json")
    write_data_readme(df, PROCESSED_DIR / "data_README.md")
    print(f"Processed {len(df)} contributions -> {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
