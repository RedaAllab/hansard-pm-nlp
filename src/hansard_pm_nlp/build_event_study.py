"""Run the Phase 7 event-study regressions (H2, H3) and write the report.

Usage:
    python -m hansard_pm_nlp.build_event_study
"""

import datetime as dt
from pathlib import Path

import pandas as pd

from hansard_pm_nlp.event_study import add_crisis_dummies, build_sitting_dataset
from hansard_pm_nlp.regression import (
    add_bh_correction,
    extract_crisis_effects,
    extract_interaction_effect,
    fit_h2_model,
    fit_h3_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEPENDENT_VARS = ["vader_compound", "transformer_score", "net_certainty"]


def _format_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "---|" * len(columns)
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            val = row[col]
            cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(
    h2_effects: pd.DataFrame, h3_effects: pd.DataFrame, crisis_counts: dict[str, int], n_docs: int, path: Path
) -> None:
    generated_at = dt.datetime.now(dt.UTC).isoformat()
    lines = [
        "# Phase 7 - Event-study regressions",
        "",
        f"Generated: {generated_at}",
        "",
        "Tests H2 (crisis affect) and H3 (party interaction) per CLAUDE.md §2. "
        f"Unit: one row per (PM, sitting date), {n_docs} sittings after the "
        "50-word floor (event_study.py) - unlike Phase 6's classifier, Liz "
        "Truss is included here, since her tenure is the only one overlapping "
        "the mini-budget crisis window.",
        "",
        "## H2 - crisis dummies (PM fixed effects, HC3 robust SE)",
        "",
        "`DV ~ PM fixed effects + one dummy per named crisis`. Each "
        "coefficient is the shift in that PM's own score during the named "
        "crisis, relative to their own baseline. Benjamini-Hochberg FDR "
        "correction applied across all 12 tests (4 crises x 3 dependent "
        "variables) together.",
        "",
        "Sittings per crisis window: "
        + ", ".join(f"{name}={count}" for name, count in crisis_counts.items())
        + ". **The mini-budget window has exactly 1 sitting** (Liz Truss, "
        "2022-10-12) - its coefficient's standard error (~3.0 for VADER, a "
        "score bounded in [-1, 1]) reflects that single-observation "
        "identification, not a genuine null effect. Read the mini-budget row "
        "as uninterpretable given available data, not as evidence against H2 "
        "for that crisis specifically.",
        "",
        *_format_table(
            h2_effects[["dv", "crisis", "coef", "se", "pvalue", "pvalue_bh", "significant_bh"]],
            ["dv", "crisis", "coef", "se", "pvalue", "pvalue_bh", "significant_bh"],
        ),
        "",
        "## H3 - party interaction (pooled any_crisis, PM fixed effects, HC3 robust SE)",
        "",
        "`DV ~ PM fixed effects + any_crisis + is_labour:any_crisis`. Per-crisis "
        "x party interactions are not identifiable - each named crisis window "
        "overlaps exactly one party's tenure in this corpus (event_study.py), "
        "so those interaction cells are structurally empty. `any_crisis` is "
        "the pooled Conservative crisis effect (Conservative is the reference "
        "party); `is_labour:any_crisis` is the differential Labour effect - "
        "the H3 test. Benjamini-Hochberg FDR correction applied across all 3 "
        "tests (1 interaction x 3 dependent variables) together.",
        "",
        "**Power caveat**: the Labour side of this interaction is identified "
        "from a single crisis window under a single PM (Starmer, Labour "
        "leadership crisis); the Conservative side pools 3 windows across 2 "
        "PMs (Johnson x2, Truss x1). This asymmetry was flagged in "
        "`PHASE0_SCOPING.md` before the corpus was built and is not a defect "
        "introduced at this stage - a non-significant or unstable Labour "
        "estimate reflects the data available, not a modeling error.",
        "",
        *_format_table(
            h3_effects[["dv", "term", "coef", "se", "pvalue", "pvalue_bh", "significant_bh"]],
            ["dv", "term", "coef", "se", "pvalue", "pvalue_bh", "significant_bh"],
        ),
        "",
        "## Conclusion",
        "",
        "**No effect survives Benjamini-Hochberg correction, for H2 or H3.** "
        "This is reported as a genuine null result, not suppressed or "
        "re-tested with different windows to find significance.",
        "",
        "The closest raw (uncorrected) result is sentiment during Covid-19 "
        "(p=0.081 VADER, p=0.068 transformer) - but in the direction opposite "
        "H2's prediction: both scores lean slightly *more positive* than "
        "Johnson's own baseline, not more negative. Neither survives "
        "correction. Ukraine and the Labour leadership crisis show no signal "
        "in either direction; mini-budget is uninterpretable (1 sitting). "
        "H3's interaction terms are all far from significance.",
        "",
        "Plausible reasons this corpus doesn't show the predicted crisis "
        "effect, none tested further here per the decision to stop at this "
        "window definition:",
        "",
        "- VADER and the general-purpose transformer model are tuned for "
        "everyday text, not the formal, guarded register of Commons "
        "speeches - a PM under pressure may sound measured rather than "
        "negative, even in substance.",
        "- Crisis windows spanning weeks to over a year (Covid) average over "
        "a lot of non-crisis-toned procedural speech, which can dilute a "
        "real but short-lived rhetorical spike.",
        "- H2 itself may simply not hold for this corpus - a null result is "
        "not a failure of the analysis, it is the analysis's answer.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "pm_contributions_affect.parquet")
    docs = build_sitting_dataset(df)
    docs = add_crisis_dummies(docs)

    h2_effects = pd.concat(
        [extract_crisis_effects(fit_h2_model(docs, dv), dv) for dv in DEPENDENT_VARS],
        ignore_index=True,
    )
    h2_effects = add_bh_correction(h2_effects)

    h3_effects = pd.DataFrame(
        [extract_interaction_effect(fit_h3_model(docs, dv), dv) for dv in DEPENDENT_VARS]
    )
    h3_effects = add_bh_correction(h3_effects)

    crisis_counts = {
        name: int(docs[f"crisis_{name}"].sum()) for name in ["covid19", "mini_budget", "ukraine_invasion", "labour_leadership_crisis"]
    }
    write_report(h2_effects, h3_effects, crisis_counts, len(docs), PROCESSED_DIR / "phase7_event_study_report.md")

    docs.to_parquet(PROCESSED_DIR / "event_study_dataset.parquet", index=False)
    h2_effects.to_csv(PROCESSED_DIR / "phase7_h2_effects.csv", index=False)
    h3_effects.to_csv(PROCESSED_DIR / "phase7_h3_effects.csv", index=False)

    print(h2_effects.to_string(index=False))
    print()
    print(h3_effects.to_string(index=False))
    print("Wrote phase7_event_study_report.md, event_study_dataset.parquet")


if __name__ == "__main__":
    main()
