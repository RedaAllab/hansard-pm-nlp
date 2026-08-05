"""Train and evaluate the PM-attribution classifier (Phase 6, H1).

Reuses the PM x sitting_date document unit from topic_corpus.py (Phase 5)
for consistency. Liz Truss is excluded (see split.py); the temporal
train/test split trains on each remaining PM's earlier sittings and tests on
their later ones, so a good score reflects genuine stylistic generalization
rather than a leaked news cycle.

Usage:
    python -m hansard_pm_nlp.build_classifier
"""

import datetime as dt
from pathlib import Path

import pandas as pd

from hansard_pm_nlp.classifier import (
    build_feature_matrix,
    build_hist_gradient_boosting,
    build_logistic_regression,
    chance_baselines,
    evaluate,
    feature_importance,
)
from hansard_pm_nlp.split import temporal_train_test_split
from hansard_pm_nlp.style_features import load_nlp
from hansard_pm_nlp.topic_corpus import build_topic_documents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TEST_SIZE = 0.2


def _format_confusion_matrix(labels: list[str], matrix: list[list[int]]) -> list[str]:
    header = "| actual \\ predicted | " + " | ".join(labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    lines = [header, sep]
    for label, row in zip(labels, matrix, strict=True):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    return lines


def write_report(
    baselines: dict[str, float],
    logreg_result: dict,
    hgb_result: dict,
    logreg_importance: list[tuple[str, float]],
    hgb_importance: list[tuple[str, float]],
    n_train: int,
    n_test: int,
    path: Path,
) -> None:
    generated_at = dt.datetime.now(dt.UTC).isoformat()
    lines = [
        "# Phase 6 - PM-attribution classifier",
        "",
        f"Generated: {generated_at}",
        "",
        "Tests H1: 'a supervised classifier can attribute an anonymized speech "
        "excerpt to the correct PM at a rate significantly above chance' "
        "(CLAUDE.md). Document unit: one document per (PM, sitting date), "
        "same as Phase 5.",
        "",
        "## Scope",
        "",
        "3 classes: Boris Johnson, Rishi Sunak, Keir Starmer. Liz Truss "
        "excluded (5 documents total, 49-day tenure - too few for a "
        "train/test split or reliable per-class metrics; see split.py).",
        "",
        f"Train: {n_train} documents (earlier sittings per PM). "
        f"Test: {n_test} documents (later sittings per PM, temporal split, "
        f"test_size={TEST_SIZE}).",
        "",
        "## Features",
        "",
        "Lexical diversity (TTR, MTLD), readability (Flesch-Kincaid, mean "
        "words/sentence), hedging/certainty rates, individual function-word "
        "frequencies, and POS-tag distribution (spaCy) - see style_features.py. "
        "Class imbalance (Johnson 144 vs Sunak 63 vs Starmer 84 documents "
        "pre-split) handled via class_weight='balanced' in both models.",
        "",
        "## Chance baselines",
        "",
        f"- Uniform random (1/3 classes): {baselines['uniform_random']:.3f}",
        f"- Always predict majority class (train set): {baselines['majority_class']:.3f}",
        "",
        "## Results",
        "",
        "| Model | Accuracy | Macro F1 |",
        "|---|---|---|",
        f"| Logistic regression | {logreg_result['accuracy']:.3f} | {logreg_result['macro_f1']:.3f} |",
        f"| HistGradientBoosting | {hgb_result['accuracy']:.3f} | {hgb_result['macro_f1']:.3f} |",
        "",
        "## Confusion matrix - logistic regression",
        "",
        *_format_confusion_matrix(logreg_result["labels"], logreg_result["confusion_matrix"]),
        "",
        "## Confusion matrix - HistGradientBoosting",
        "",
        *_format_confusion_matrix(hgb_result["labels"], hgb_result["confusion_matrix"]),
        "",
        "## Feature importance (permutation, top 15)",
        "",
        "Model-agnostic (permutation_importance on the test set), so both "
        "models are ranked the same way rather than comparing a linear "
        "model's coefficients to a tree ensemble's split gains.",
        "",
        "### Logistic regression",
        "",
    ]
    for name, score in logreg_importance:
        lines.append(f"- {name}: {score:.4f}")
    lines += ["", "### HistGradientBoosting", ""]
    for name, score in hgb_importance:
        lines.append(f"- {name}: {score:.4f}")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "pm_contributions_clean.parquet")
    docs = build_topic_documents(df)

    train_docs, test_docs = temporal_train_test_split(docs, test_size=TEST_SIZE)

    nlp = load_nlp()
    train_features = build_feature_matrix(train_docs, nlp)
    test_features = build_feature_matrix(test_docs, nlp)

    X_train, y_train = train_features.drop(columns=["pm_name"]), train_features["pm_name"]
    X_test, y_test = test_features.drop(columns=["pm_name"]), test_features["pm_name"]

    logreg = build_logistic_regression()
    logreg.fit(X_train, y_train)
    logreg_result = evaluate(logreg, X_test, y_test)
    logreg_importance = feature_importance(logreg, X_test, y_test)

    hgb = build_hist_gradient_boosting()
    hgb.fit(X_train, y_train)
    hgb_result = evaluate(hgb, X_test, y_test)
    hgb_importance = feature_importance(hgb, X_test, y_test)

    baselines = chance_baselines(y_train, y_test)

    write_report(
        baselines,
        logreg_result,
        hgb_result,
        logreg_importance,
        hgb_importance,
        len(train_docs),
        len(test_docs),
        PROCESSED_DIR / "phase6_classifier_report.md",
    )

    all_features = pd.concat([train_features.assign(split="train"), test_features.assign(split="test")])
    all_features = pd.concat(
        [pd.concat([train_docs, test_docs]).reset_index(drop=True)[["sitting_date"]], all_features.reset_index(drop=True)],
        axis=1,
    )
    all_features.to_parquet(PROCESSED_DIR / "pm_style_features.parquet", index=False)

    # Dashboard (Phase 8) reads these directly rather than retraining sklearn
    # models on every page load.
    predictions = test_docs[["pm_name", "sitting_date"]].reset_index(drop=True)
    predictions["pred_logreg"] = logreg.predict(X_test)
    predictions["pred_hgb"] = hgb.predict(X_test)
    predictions.to_csv(PROCESSED_DIR / "phase6_test_predictions.csv", index=False)

    pd.DataFrame(logreg_importance, columns=["feature", "importance"]).to_csv(
        PROCESSED_DIR / "phase6_feature_importance_logreg.csv", index=False
    )
    pd.DataFrame(hgb_importance, columns=["feature", "importance"]).to_csv(
        PROCESSED_DIR / "phase6_feature_importance_hgb.csv", index=False
    )
    pd.DataFrame(
        [
            {"model": "logreg", **{k: v for k, v in logreg_result.items() if k in ("accuracy", "macro_f1")}},
            {"model": "hgb", **{k: v for k, v in hgb_result.items() if k in ("accuracy", "macro_f1")}},
        ]
    ).to_csv(PROCESSED_DIR / "phase6_metrics.csv", index=False)

    print(f"Logistic regression: accuracy={logreg_result['accuracy']:.3f}, macro_f1={logreg_result['macro_f1']:.3f}")
    print(f"HistGradientBoosting: accuracy={hgb_result['accuracy']:.3f}, macro_f1={hgb_result['macro_f1']:.3f}")
    print(f"Chance baselines: {baselines}")
    print("Wrote phase6_classifier_report.md, pm_style_features.parquet, phase6_test_predictions.csv")


if __name__ == "__main__":
    main()
