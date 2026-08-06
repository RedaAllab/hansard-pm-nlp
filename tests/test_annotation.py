import pandas as pd
import pytest

from hansard_pm_nlp.annotation import (
    LABEL_COLUMN,
    MIN_WORDS,
    build_sample,
    export_template,
    load_annotations,
)

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
