"""Hand-annotation sample for the fine-tuned sentiment model.

Builds a stratified sample, writes a blank CSV for a human coder to fill in,
and reads it back with validation. Everything here is infrastructure - **the
labels themselves are not produced by this module and must not be.** The
point of this sample is to be an independent ground truth for fine-tuning and
evaluating a sentiment classifier; generating it automatically would defeat
that purpose (see PROJECT_SUMMARY.md's "Long-term evolutions" entry on the
fine-tuned sentiment model).

Sampling is stratified by PM, disproportionately: Liz Truss's 123 contributions
are 1% of the corpus, so a proportional draw would give her ~2-3 rows, too few
to say anything about how the model treats her register. She is oversampled
relative to her true share instead - the same reasoning `split.py::EXCLUDED_PMS`
already documents for excluding her from the PM classifier entirely, applied
here as oversampling rather than exclusion since the sentiment task isn't
PM-conditional.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hansard_pm_nlp.event_study import CRISIS_WINDOWS
from hansard_pm_nlp.lexical import tokenize_words

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SAMPLE_FILENAME = "sentiment_annotation_sample.csv"
GUIDELINES_FILENAME = "annotation_guidelines.md"

#: Fixed so the sample is reproducible - re-running before annotation starts
#: must not silently swap out which contributions were drawn.
RANDOM_SEED = 20260806

#: Per-PM sample sizes, agreed with the project owner: proportional to corpus
#: share for Johnson/Sunak/Starmer, oversampled for Truss (see module
#: docstring). Total: 250.
PM_SAMPLE_TARGETS = {
    "Boris Johnson": 105,
    "Rishi Sunak": 55,
    "Keir Starmer": 70,
    "Liz Truss": 20,
}

#: A random PM-stratified draw barely touches the crisis windows CRISIS_WINDOWS
#: names (mini_budget is 25 days, ukraine_invasion 3 months, out of years-long
#: tenures) - the first 250-row sample landed only 5/9/3 rows in mini_budget/
#: ukraine_invasion/labour_leadership_crisis. That's fine for evaluating raw
#: classifier accuracy, but useless for the actual downstream question (does a
#: fine-tuned score change H2/H3's null result), which needs enough gold labels
#: *inside* those windows to check. covid19 already has 37 rows from the first
#: draw (16-month window), so it is not topped up here. Seed offset from
#: RANDOM_SEED so this second draw doesn't reproduce the same row order.
CRISIS_SAMPLE_SEED = RANDOM_SEED + 1
CRISIS_SAMPLE_TARGETS = {
    "mini_budget": 15,
    "ukraine_invasion": 15,
    "labour_leadership_crisis": 15,
}

#: Contributions shorter than this carry too little text to judge tone from -
#: single-word interjections ("No.", "Order.") dominate the short tail of the
#: corpus and would waste annotation time. Same role as trans_framing's
#: MIN_TOKENS, threshold picked fresh for this corpus's own length profile.
MIN_WORDS = 20

#: The label a human coder assigns. A plain string column, not one-hot -
#: sentiment here is a single mutually-exclusive judgement, unlike the
#: trans_framing project's non-exclusive frames.
LABEL_COLUMN = "sentiment_label"
VALID_LABELS = ("positive", "neutral", "negative")

CONTEXT_COLUMNS = (
    "contribution_ext_id",
    "pm_name",
    "sitting_date",
    "debate_section",
    "is_pmqs",
    "contribution_text",
)

#: Shown on the annotation screen next to each label, and rendered into
#: GUIDELINES_FILENAME - single source of truth so the on-screen text and the
#: committed doc can't drift apart.
LABEL_DEFINITIONS = {
    "positive": (
        "Réassurance, confiance, éloge",
        "optimisme affiché, remerciements, fierté, mise en avant d'un succès "
        "ou d'une bonne nouvelle",
    ),
    "neutral": (
        "Procédural / informatif, sans valence affective nette",
        "réponse factuelle, annonce de calendrier, rappel de chiffres ou de "
        "procédure, renvoi vers un autre ministre sans jugement de valeur",
    ),
    "negative": (
        "Critique, inquiétude, reproche",
        "mise en cause d'un adversaire ou d'une situation, alerte sur un "
        "problème, ton défensif ou combatif, déploration",
    ),
}

DECISION_RULES = (
    "Juger le ton dominant de l'ensemble de la contribution, pas un mot isolé.",
    "En cas de mélange sans dominante claire, choisir le pôle le plus appuyé "
    "plutôt que de forcer un compromis - Neutre est réservé aux contributions "
    "sans contenu évaluatif du tout, pas aux contributions mitigées.",
    "L'ironie ou le sarcasme se codent selon le ton réel visé (souvent "
    "négatif/critique), pas selon le sens littéral des mots.",
    "Une déviation vers l'opposition sans reproche explicite reste Neutre ; "
    "avec un reproche explicite, elle devient Négatif.",
    "On code le sentiment exprimé PAR le Premier ministre, jamais le "
    "sentiment des autres à son sujet.",
    "En cas de mélange où un pôle est porté par des affirmations concrètes "
    "ou chiffrées et l'autre par une formule générique, privilégier le pôle "
    "concret - ex. un bilan chiffré pèse plus qu'un slogan de réassurance "
    "générique comme 'nous allons régler ça'.",
    "Si la contribution répond visiblement à une question absente du texte "
    "(le corpus ne contient que les tours du Premier ministre), coder "
    "uniquement son contenu explicite visible, sans deviner la question.",
)


def _eligible(contributions: pd.DataFrame) -> pd.DataFrame:
    lengths = contributions["contribution_text"].map(lambda t: len(tokenize_words(t)))
    return contributions[lengths >= MIN_WORDS]


def build_sample(
    contributions: pd.DataFrame,
    targets: dict[str, int] = PM_SAMPLE_TARGETS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Draw the stratified annotation sample from the cleaned corpus.

    Raises if a PM's eligible pool is smaller than its target, rather than
    silently under-sampling - that would quietly shrink the agreed-upon
    stratification without anyone noticing.
    """
    eligible = _eligible(contributions)

    parts = []
    for pm_name, target in targets.items():
        pool = eligible[eligible["pm_name"] == pm_name]
        if len(pool) < target:
            raise ValueError(
                f"{pm_name} has only {len(pool)} eligible contributions "
                f"(>= {MIN_WORDS} words), fewer than the target of {target}."
            )
        parts.append(pool.sample(n=target, random_state=seed))

    sample = pd.concat(parts).sort_values(["pm_name", "sitting_date"])
    return sample.reset_index(drop=True)


def build_crisis_sample(
    contributions: pd.DataFrame,
    exclude_ids: pd.Series | list[str],
    targets: dict[str, int] = CRISIS_SAMPLE_TARGETS,
    seed: int = CRISIS_SAMPLE_SEED,
) -> pd.DataFrame:
    """Draw a supplementary sample from inside named crisis windows.

    `exclude_ids` must be the contribution_ext_ids already in the main sample -
    otherwise a contribution could be drawn twice across the two files, and a
    human coder would silently label the same text under two different rows.

    Raises on a pool smaller than its target, same reasoning as build_sample:
    mini_budget in particular only has ~23 eligible contributions total (a
    25-day window, one PM), so silently under-filling would be easy to miss.
    """
    eligible = _eligible(contributions)
    eligible = eligible[~eligible["contribution_ext_id"].isin(set(exclude_ids))]

    parts = []
    seen_ids: set[str] = set()
    for window_name, target in targets.items():
        start, end = CRISIS_WINDOWS[window_name]
        in_window = eligible["sitting_date"].astype(str).between(start, end)
        pool = eligible[in_window & ~eligible["contribution_ext_id"].isin(seen_ids)]
        if len(pool) < target:
            raise ValueError(
                f"Crisis window '{window_name}' has only {len(pool)} eligible "
                f"contributions (>= {MIN_WORDS} words, not already sampled), "
                f"fewer than the target of {target}."
            )
        drawn = pool.sample(n=target, random_state=seed)
        seen_ids |= set(drawn["contribution_ext_id"])
        parts.append(drawn)

    sample = pd.concat(parts).sort_values("sitting_date")
    return sample.reset_index(drop=True)


def append_to_sample(addition: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Add blank-labeled rows to an existing sample CSV, preserving its labels.

    Unlike export_template, this is meant to run against a file that may
    already have annotations in it - it must never touch existing rows, only
    add new ones. Raises on an id already present, so a re-run can't silently
    duplicate a row a coder has already labeled.
    """
    resolved = Path(path) if path is not None else PROCESSED_DIR / SAMPLE_FILENAME
    existing = pd.read_csv(resolved, dtype={LABEL_COLUMN: "string"})

    overlap = set(existing["contribution_ext_id"]) & set(addition["contribution_ext_id"])
    if overlap:
        raise ValueError(f"{len(overlap)} contribution(s) already in {resolved}: {overlap}")

    new_rows = addition[list(CONTEXT_COLUMNS)].copy()
    new_rows[LABEL_COLUMN] = ""
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined.to_csv(resolved, index=False)
    return resolved


def export_template(sample: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Write the blank annotation template, refusing to clobber a filled one.

    Overwriting a partially annotated file would destroy hours of manual work
    with no way back, so an existing file is an error rather than a warning.
    """
    resolved = Path(path) if path is not None else PROCESSED_DIR / SAMPLE_FILENAME
    if resolved.exists():
        raise FileExistsError(
            f"{resolved} already exists - refusing to overwrite possible annotations. "
            "Delete it explicitly if you really want a fresh sample."
        )

    template = sample[list(CONTEXT_COLUMNS)].copy()
    template[LABEL_COLUMN] = ""
    resolved.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(resolved, index=False)
    return resolved


def load_annotations(path: str | Path | None = None) -> pd.DataFrame:
    """Read a filled annotation file, validating that it really is filled.

    Raises rather than silently returning a partial sample: fine-tuning or
    evaluating against an incomplete/malformed label set would produce
    numbers that look real but rest on an arbitrary subset.
    """
    resolved = Path(path) if path is not None else PROCESSED_DIR / SAMPLE_FILENAME
    df = pd.read_csv(resolved)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"{resolved} is missing the '{LABEL_COLUMN}' column.")

    blank = df[LABEL_COLUMN].isna() | (df[LABEL_COLUMN].astype(str).str.strip() == "")
    if blank.any():
        raise ValueError(
            f"{resolved} has {int(blank.sum())} unannotated rows out of {len(df)}. "
            f"Fill every '{LABEL_COLUMN}' cell with one of {VALID_LABELS} before continuing."
        )

    invalid = ~df[LABEL_COLUMN].isin(VALID_LABELS)
    if invalid.any():
        bad_values = sorted(df.loc[invalid, LABEL_COLUMN].unique())
        raise ValueError(
            f"{resolved} has label values outside {VALID_LABELS}: {bad_values}"
        )

    return df


def write_guidelines(path: str | Path | None = None) -> Path:
    """Render LABEL_DEFINITIONS/DECISION_RULES to a standalone markdown doc.

    Generated from the same constants the HTML tool reads (annotation_tool.py),
    so the committed reference doc can't drift from what the coder actually
    sees on screen.
    """
    resolved = Path(path) if path is not None else PROJECT_ROOT / GUIDELINES_FILENAME

    lines = [
        "# Sentiment annotation guidelines",
        "",
        "Reference for hand-labeling `data/processed/sentiment_annotation_sample.csv` "
        "(see `annotation.py`'s module docstring for why this sample exists and how "
        "it was drawn). The same definitions and rules are shown inline in "
        "`sentiment_annotation_tool.html` at the moment of each decision - this file "
        "is the durable, portfolio-facing copy.",
        "",
        "## Labels",
        "",
    ]
    for label in VALID_LABELS:
        title, examples = LABEL_DEFINITIONS[label]
        lines += [f"**{label.capitalize()}** - {title}", "", f"_{examples}_", ""]

    lines += ["## Decision rules", ""]
    lines += [f"- {rule}" for rule in DECISION_RULES]
    lines.append("")

    resolved.write_text("\n".join(lines), encoding="utf-8")
    return resolved


def main() -> None:
    path = PROCESSED_DIR / SAMPLE_FILENAME
    guidelines_path = PROJECT_ROOT / GUIDELINES_FILENAME

    if not path.exists():
        contributions = pd.read_parquet(PROCESSED_DIR / "pm_contributions_clean.parquet")
        sample = build_sample(contributions)
        export_template(sample, path)
        written_guidelines = write_guidelines(guidelines_path)
        print(f"Wrote a blank {len(sample)}-contribution template to {path}")
        for pm_name, target in PM_SAMPLE_TARGETS.items():
            print(f"  {pm_name}: {target}")
        print(f"Wrote {written_guidelines}")
        print(f"\nAnnotate the '{LABEL_COLUMN}' column by hand with one of {VALID_LABELS},")
        print("or open sentiment_annotation_tool.html for a friendlier interface")
        print("(python -m hansard_pm_nlp.annotation_tool).")
        return

    try:
        annotations = load_annotations(path)
    except ValueError as exc:
        print(f"Not fully annotated yet: {exc}")
        return

    print(f"{path} is fully annotated: {len(annotations)} labeled contributions.")
    print(annotations[LABEL_COLUMN].value_counts().to_string())


if __name__ == "__main__":
    main()
