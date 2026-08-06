import pandas as pd
import pytest

from hansard_pm_nlp.annotation import (
    CONTEXT_COLUMNS,
    LABEL_COLUMN,
    MIN_WORDS,
    append_to_sample,
    build_crisis_sample,
    build_sample,
    export_template,
    load_annotations,
)
from hansard_pm_nlp.event_study import CRISIS_WINDOWS

LONG_TEXT = " ".join(["word"] * (MIN_WORDS + 5))
SHORT_TEXT = "No."


def _contributions(pm_counts: dict[str, int], text: str = LONG_TEXT) -> pd.DataFrame:
    rows = []
    i = 0
    for pm_name, n in pm_counts.items():
        for _ in range(n):
            rows.append(
                {
                    "contribution_ext_id": f"c{i}",
                    "pm_name": pm_name,
                    "sitting_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "debate_section": "Engagements",
                    "is_pmqs": True,
                    "contribution_text": text,
                }
            )
            i += 1
    return pd.DataFrame(rows)


def _crisis_contributions(window_counts: dict[str, int], text: str = LONG_TEXT) -> pd.DataFrame:
    """Rows dated inside real CRISIS_WINDOWS ranges, for build_crisis_sample tests."""
    rows = []
    i = 0
    for window_name, n in window_counts.items():
        start = pd.Timestamp(CRISIS_WINDOWS[window_name][0])
        for _ in range(n):
            rows.append(
                {
                    "contribution_ext_id": f"x{i}",
                    "pm_name": "Boris Johnson",
                    "sitting_date": start + pd.Timedelta(days=i % 3),
                    "debate_section": "Engagements",
                    "is_pmqs": True,
                    "contribution_text": text,
                }
            )
            i += 1
    return pd.DataFrame(rows)


def _targets(pm_counts: dict[str, int]) -> dict[str, int]:
    return {pm: n for pm, n in pm_counts.items()}


def test_sample_matches_per_pm_targets():
    contributions = _contributions({"Boris Johnson": 20, "Liz Truss": 10})
    targets = {"Boris Johnson": 8, "Liz Truss": 5}
    sample = build_sample(contributions, targets=targets)
    counts = sample["pm_name"].value_counts().to_dict()
    assert counts == targets


def test_sample_is_reproducible_across_runs():
    contributions = _contributions({"Boris Johnson": 20})
    targets = {"Boris Johnson": 8}
    first = build_sample(contributions, targets=targets)["contribution_ext_id"].tolist()
    second = build_sample(contributions, targets=targets)["contribution_ext_id"].tolist()
    assert first == second


def test_sample_draws_distinct_contributions():
    contributions = _contributions({"Boris Johnson": 20})
    sample = build_sample(contributions, targets={"Boris Johnson": 8})
    assert sample["contribution_ext_id"].is_unique


def test_short_contributions_are_excluded_before_sampling():
    contributions = _contributions({"Boris Johnson": 5}, text=SHORT_TEXT)
    with pytest.raises(ValueError, match="fewer than the target"):
        build_sample(contributions, targets={"Boris Johnson": 3})


def test_raises_when_pool_smaller_than_target():
    contributions = _contributions({"Boris Johnson": 3})
    with pytest.raises(ValueError, match="fewer than the target"):
        build_sample(contributions, targets={"Boris Johnson": 5})


def test_export_template_refuses_to_overwrite(tmp_path):
    contributions = _contributions({"Boris Johnson": 5})
    sample = build_sample(contributions, targets={"Boris Johnson": 3})
    path = tmp_path / "sample.csv"
    export_template(sample, path)
    with pytest.raises(FileExistsError):
        export_template(sample, path)


def test_export_template_has_blank_label_column(tmp_path):
    contributions = _contributions({"Boris Johnson": 5})
    sample = build_sample(contributions, targets={"Boris Johnson": 3})
    path = tmp_path / "sample.csv"
    export_template(sample, path)
    written = pd.read_csv(path)
    assert written[LABEL_COLUMN].isna().all()


def test_load_annotations_rejects_blank_rows(tmp_path):
    df = pd.DataFrame(
        {"contribution_ext_id": ["c0", "c1"], LABEL_COLUMN: ["positive", ""]}
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unannotated"):
        load_annotations(path)


def test_load_annotations_rejects_invalid_labels(tmp_path):
    df = pd.DataFrame(
        {"contribution_ext_id": ["c0", "c1"], LABEL_COLUMN: ["positive", "very_positive"]}
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="outside"):
        load_annotations(path)


def test_load_annotations_accepts_fully_labeled_file(tmp_path):
    df = pd.DataFrame(
        {"contribution_ext_id": ["c0", "c1"], LABEL_COLUMN: ["positive", "negative"]}
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    loaded = load_annotations(path)
    assert list(loaded[LABEL_COLUMN]) == ["positive", "negative"]


def test_crisis_sample_matches_targets():
    contributions = _crisis_contributions({"mini_budget": 10, "ukraine_invasion": 10})
    targets = {"mini_budget": 4, "ukraine_invasion": 6}
    sample = build_crisis_sample(contributions, exclude_ids=[], targets=targets)
    assert len(sample) == 10
    for window_name, target in targets.items():
        start, end = CRISIS_WINDOWS[window_name]
        n_in_window = sample["sitting_date"].astype(str).between(start, end).sum()
        assert n_in_window == target


def test_crisis_sample_excludes_already_sampled_ids():
    contributions = _crisis_contributions({"mini_budget": 5})
    already = contributions["contribution_ext_id"].iloc[:2].tolist()
    sample = build_crisis_sample(contributions, exclude_ids=already, targets={"mini_budget": 3})
    assert not set(sample["contribution_ext_id"]) & set(already)


def test_crisis_sample_raises_when_window_pool_too_small():
    contributions = _crisis_contributions({"mini_budget": 2})
    with pytest.raises(ValueError, match="fewer than the target"):
        build_crisis_sample(contributions, exclude_ids=[], targets={"mini_budget": 5})


def test_append_to_sample_preserves_existing_labels(tmp_path):
    path = tmp_path / "sample.csv"
    existing = pd.DataFrame(
        {**{col: ["v"] for col in CONTEXT_COLUMNS}, LABEL_COLUMN: ["positive"]}
    )
    existing["contribution_ext_id"] = ["c0"]
    existing.to_csv(path, index=False)

    addition = pd.DataFrame(
        {**{col: ["v"] for col in CONTEXT_COLUMNS}}
    )
    addition["contribution_ext_id"] = ["c1"]

    append_to_sample(addition, path)
    combined = pd.read_csv(path)
    assert len(combined) == 2
    assert combined.loc[combined["contribution_ext_id"] == "c0", LABEL_COLUMN].iloc[0] == "positive"
    assert pd.isna(combined.loc[combined["contribution_ext_id"] == "c1", LABEL_COLUMN].iloc[0])


def test_append_to_sample_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "sample.csv"
    existing = pd.DataFrame(
        {**{col: ["v"] for col in CONTEXT_COLUMNS}, LABEL_COLUMN: [""]}
    )
    existing["contribution_ext_id"] = ["c0"]
    existing.to_csv(path, index=False)

    addition = pd.DataFrame({**{col: ["v"] for col in CONTEXT_COLUMNS}})
    addition["contribution_ext_id"] = ["c0"]

    with pytest.raises(ValueError, match="already in"):
        append_to_sample(addition, path)
