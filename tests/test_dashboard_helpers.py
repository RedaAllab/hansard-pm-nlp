import pandas as pd

from hansard_pm_nlp.dashboard_helpers import (
    confusion_cell_detail,
    crisis_baseline_split,
    crisis_party_split,
    normalize_radar,
    parse_tfidf_terms,
)


def _toy_profile():
    return pd.DataFrame(
        {
            "pm_name": ["A", "B", "C"],
            "metric_x": [0.0, 5.0, 10.0],
            "metric_y": [1.0, 1.0, 1.0],
        }
    )


def test_normalize_radar_maps_min_and_max_to_0_and_1():
    normalized = normalize_radar(_toy_profile(), ["metric_x"], ["A", "B", "C"])
    assert normalized.loc["A", "metric_x"] == 0.0
    assert normalized.loc["C", "metric_x"] == 1.0
    assert normalized.loc["B", "metric_x"] == 0.5


def test_normalize_radar_centers_tied_column_at_half():
    normalized = normalize_radar(_toy_profile(), ["metric_y"], ["A", "B", "C"])
    assert (normalized["metric_y"] == 0.5).all()


def test_normalize_radar_single_pm_selected_is_all_half():
    normalized = normalize_radar(_toy_profile(), ["metric_x", "metric_y"], ["B"])
    assert (normalized.loc["B"] == 0.5).all()


def test_normalize_radar_rescales_when_outlier_deselected():
    # With C (the outlier) excluded, A and B become each other's min/max.
    normalized = normalize_radar(_toy_profile(), ["metric_x"], ["A", "B"])
    assert normalized.loc["A", "metric_x"] == 0.0
    assert normalized.loc["B", "metric_x"] == 1.0


def _toy_predictions():
    return pd.DataFrame(
        {
            "pm_name": ["A", "A", "B", "B"],
            "sitting_date": ["2020-01-01", "2020-01-02", "2020-02-01", "2020-02-02"],
            "pred_logreg": ["A", "B", "B", "B"],
        }
    )


def test_confusion_cell_detail_returns_matching_rows_only():
    detail = confusion_cell_detail(_toy_predictions(), "pred_logreg", "A", "B")
    assert detail["sitting_date"].tolist() == ["2020-01-02"]


def test_confusion_cell_detail_diagonal_cell():
    detail = confusion_cell_detail(_toy_predictions(), "pred_logreg", "B", "B")
    assert detail["sitting_date"].tolist() == ["2020-02-01", "2020-02-02"]


def test_confusion_cell_detail_empty_cell():
    detail = confusion_cell_detail(_toy_predictions(), "pred_logreg", "B", "A")
    assert detail.empty


def test_parse_tfidf_terms_preserves_order_and_scores():
    pairs = parse_tfidf_terms("hon (0.485); right (0.325); people (0.268)")
    assert pairs == [("hon", 0.485), ("right", 0.325), ("people", 0.268)]


def test_parse_tfidf_terms_empty_string():
    assert parse_tfidf_terms("") == []


def _toy_event_df():
    return pd.DataFrame(
        {
            "pm_party": ["Conservative", "Conservative", "Labour", "Labour"],
            "crisis_covid19": [True, False, False, False],
            "any_crisis": [True, False, True, False],
            "vader_compound": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_crisis_baseline_split_labels_periods():
    split = crisis_baseline_split(_toy_event_df(), "crisis_covid19", "vader_compound")
    assert split["period"].tolist() == ["Crisis", "Baseline", "Baseline", "Baseline"]
    assert split["value"].tolist() == [0.1, 0.2, 0.3, 0.4]


def test_crisis_party_split_labels_party_and_period():
    split = crisis_party_split(_toy_event_df(), "vader_compound")
    assert split["party"].tolist() == ["Conservative", "Conservative", "Labour", "Labour"]
    assert split["period"].tolist() == ["Crisis", "Baseline", "Crisis", "Baseline"]
