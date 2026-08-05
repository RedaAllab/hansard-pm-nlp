import pandas as pd

from hansard_pm_nlp.classifier import (
    build_hist_gradient_boosting,
    build_logistic_regression,
    chance_baselines,
    evaluate,
    feature_importance,
)


def _toy_features():
    # Two clearly separable classes on a couple of engineered columns, so a
    # trained classifier should do far better than chance without needing
    # spaCy or real text. HistGradientBoostingClassifier's default
    # min_samples_leaf=20 needs a train set well above that per class.
    rows = []
    for i in range(60):
        rows.append({"a": 1.0 + 0.01 * i, "b": 0.0, "pm_name": "PM A"})
    for i in range(60):
        rows.append({"a": 0.0, "b": 1.0 + 0.01 * i, "pm_name": "PM B"})
    return pd.DataFrame(rows)


def _split(features):
    train = pd.concat([features.iloc[:50], features.iloc[60:110]])
    test = pd.concat([features.iloc[50:60], features.iloc[110:120]])
    X_train, y_train = train.drop(columns=["pm_name"]), train["pm_name"]
    X_test, y_test = test.drop(columns=["pm_name"]), test["pm_name"]
    return X_train, y_train, X_test, y_test


def test_logistic_regression_beats_chance_on_separable_data():
    X_train, y_train, X_test, y_test = _split(_toy_features())
    model = build_logistic_regression()
    model.fit(X_train, y_train)
    result = evaluate(model, X_test, y_test)
    assert result["accuracy"] > 0.9


def test_hist_gradient_boosting_beats_chance_on_separable_data():
    X_train, y_train, X_test, y_test = _split(_toy_features())
    model = build_hist_gradient_boosting()
    model.fit(X_train, y_train)
    result = evaluate(model, X_test, y_test)
    assert result["accuracy"] > 0.9


def test_chance_baselines_uniform_random_matches_class_count():
    y_train = pd.Series(["PM A"] * 6 + ["PM B"] * 4 + ["PM C"] * 2)
    y_test = pd.Series(["PM A"] * 5)
    baselines = chance_baselines(y_train, y_test)
    assert baselines["uniform_random"] == 1 / 3
    assert baselines["majority_class"] == 1.0


def test_feature_importance_ranks_informative_column_first():
    X_train, y_train, X_test, y_test = _split(_toy_features())
    model = build_logistic_regression()
    model.fit(X_train, y_train)
    importances = feature_importance(model, X_test, y_test, top_k=2)
    top_feature = importances[0][0]
    assert top_feature in ("a", "b")
