"""PM-attribution classifier (Phase 6, H1): "a supervised classifier can
attribute an anonymized speech excerpt to the correct PM at a rate
significantly above chance" (CLAUDE.md, H1).

Two models, both class_weight='balanced' given Johnson's 144 documents vs
Sunak's 63 and Starmer's 84 (see split.py for why Truss is excluded
entirely): logistic regression (linear baseline, directly interpretable
coefficients) and HistGradientBoostingClassifier (non-linear, evaluated via
permutation importance since it has no built-in feature_importances_).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hansard_pm_nlp.style_features import build_style_features, load_nlp

RANDOM_SEED = 42


def build_feature_matrix(docs: pd.DataFrame, nlp=None) -> pd.DataFrame:
    """One row per document: all style_features.py columns plus pm_name."""
    nlp = nlp or load_nlp()
    rows = [build_style_features(text, nlp) for text in docs["text"]]
    features = pd.DataFrame(rows)
    features["pm_name"] = docs["pm_name"].to_numpy()
    return features


def _xy(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = features.drop(columns=["pm_name"])
    y = features["pm_name"]
    return X, y


def build_logistic_regression() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED
                ),
            ),
        ]
    )


def build_hist_gradient_boosting() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    class_weight="balanced", random_state=RANDOM_SEED
                ),
            ),
        ]
    )


def chance_baselines(y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    """Two baselines for 'significantly above chance': uniform random guess
    among classes, and always predicting the training set's majority class.
    """
    n_classes = y_train.nunique()
    majority_class = y_train.value_counts().idxmax()
    majority_accuracy = (y_test == majority_class).mean()
    return {
        "uniform_random": 1 / n_classes,
        "majority_class": float(majority_accuracy),
    }


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    labels = sorted(y_test.unique())
    return {
        "accuracy": float((y_pred == y_test).mean()),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
    }


def feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series, top_k: int = 15) -> list[tuple[str, float]]:
    """Permutation importance (model-agnostic) rather than model-specific
    attributes, so logistic regression and gradient boosting are compared
    the same way.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=RANDOM_SEED, scoring="f1_macro"
    )
    order = np.argsort(result.importances_mean)[::-1][:top_k]
    return [(X_test.columns[i], float(result.importances_mean[i])) for i in order]
