from pathlib import Path

import polars as pl
import pytest

from trainer.src.config.root import PredictionConfig
from trainer.src.pipelines.predict import (
    _derive_major_output_path,
    _derive_shard_output_path,
    _derive_sub_output_path,
    _effective_parallelism,
    _get_major_input_paths,
    _get_major_intermediate_paths,
    _get_prediction_input_paths,
    _get_sub_input_paths,
)


def test_get_prediction_input_paths_prefers_input_paths():
    cfg = PredictionConfig(
        input_path=Path("data/fallback.parquet"),
        input_paths=[Path("data/raw_a.parquet"), Path("data/raw_b.parquet")],
    )

    assert _get_prediction_input_paths(cfg) == [
        Path("data/raw_a.parquet"),
        Path("data/raw_b.parquet"),
    ]


def test_get_prediction_input_paths_falls_back_to_single_input():
    cfg = PredictionConfig(input_path=Path("data/raw.parquet"))

    assert _get_prediction_input_paths(cfg) == [Path("data/raw.parquet")]


def test_get_prediction_input_paths_filters_directory_by_required_schema(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    valid_path = raw_dir / "valid.parquet"
    invalid_path = raw_dir / "invalid.parquet"

    pl.DataFrame(
        {
            "datetime": ["2024-01-01"],
            "title": ["a"],
            "content": ["b"],
        }
    ).write_parquet(valid_path)
    pl.DataFrame({"foo": [1], "bar": [2]}).write_parquet(invalid_path)

    cfg = PredictionConfig(input_dir=raw_dir)

    assert _get_prediction_input_paths(cfg) == [valid_path]


def test_get_major_input_paths_prefers_major_input_dir(tmp_path: Path):
    raw_dir = tmp_path / "major_raw"
    raw_dir.mkdir()
    valid_path = raw_dir / "major_a.parquet"
    pl.DataFrame(
        {
            "datetime": ["2024-01-01"],
            "title": ["a"],
            "content": ["b"],
        }
    ).write_parquet(valid_path)

    cfg = PredictionConfig(
        major_input_dir=raw_dir,
        input_dir=tmp_path / "legacy",
    )

    assert _get_major_input_paths(cfg) == [valid_path]


def test_get_sub_input_paths_prefers_explicit_sub_input_dir(tmp_path: Path):
    sub_dir = tmp_path / "sub_input"
    sub_dir.mkdir()
    valid_path = sub_dir / "cached_major.parquet"
    pl.DataFrame(
        {
            "datetime": ["2024-01-01"],
            "title": ["a"],
            "content": ["b"],
            "major_category": ["科技信息"],
            "sentiment": ["positive"],
        }
    ).write_parquet(valid_path)

    major_dir = tmp_path / "major_only"
    major_dir.mkdir()
    (major_dir / "fallback_major_only.parquet").write_bytes(b"")

    cfg = PredictionConfig(
        sub_input_dir=sub_dir,
        sub_input_glob="*.parquet",
        major_output_dir=major_dir,
    )

    assert _get_sub_input_paths(cfg) == [valid_path]


def test_get_prediction_input_paths_requires_any_input():
    cfg = PredictionConfig()

    with pytest.raises(AssertionError, match="input_path or input_paths must be set"):
        _get_prediction_input_paths(cfg)


def test_derive_shard_output_path_uses_configured_output_for_single_input():
    output_path = _derive_shard_output_path(
        input_path=Path("data/raw/news_a.parquet"),
        configured_output=Path("outputs/predictions.parquet"),
        suffix="_major_only",
        total_inputs=1,
    )

    assert output_path == Path("outputs/predictions.parquet")


def test_derive_shard_output_path_uses_input_name_for_multi_input_file_output():
    output_path = _derive_shard_output_path(
        input_path=Path("data/raw/news_a.parquet"),
        configured_output=Path("outputs/predictions.parquet"),
        suffix="_major_only",
        total_inputs=2,
    )

    assert output_path == Path("outputs/news_a_major_only.parquet")


def test_derive_shard_output_path_uses_output_dir_when_configured_output_is_directory():
    output_path = _derive_shard_output_path(
        input_path=Path("data/raw/news_b.parquet"),
        configured_output=Path("outputs/shards"),
        suffix="_sub",
        total_inputs=2,
    )

    assert output_path == Path("outputs/shards/news_b_sub.parquet")


def test_derive_shard_output_path_defaults_next_to_input_when_output_not_configured():
    output_path = _derive_shard_output_path(
        input_path=Path("data/raw/news_c.parquet"),
        configured_output=None,
        suffix="_major_only",
        total_inputs=2,
    )

    assert output_path == Path("data/raw/news_c_major_only.parquet")


def test_derive_major_output_path_prefers_output_dir():
    cfg = PredictionConfig(major_output_dir=Path("outputs/major"))

    output_path = _derive_major_output_path(Path("data/raw/news_a.parquet"), cfg, total_inputs=3)

    assert output_path == Path("outputs/major/news_a_major_only.parquet")


def test_get_major_intermediate_paths_reads_major_output_dir(tmp_path: Path):
    major_dir = tmp_path / "major"
    major_dir.mkdir()
    a = major_dir / "a_major_only.parquet"
    b = major_dir / "b_major_only.parquet"
    c = major_dir / "ignore.parquet"
    for path in [a, b, c]:
        path.write_bytes(b"")

    cfg = PredictionConfig(major_output_dir=major_dir)

    assert _get_major_intermediate_paths(cfg) == [a, b]


def test_derive_sub_output_path_prefers_output_dir():
    cfg = PredictionConfig(output_dir=Path("outputs/sub"))

    output_path = _derive_sub_output_path(Path("outputs/major/news_a_major_only.parquet"), cfg, total_inputs=2)

    assert output_path == Path("outputs/sub/news_a_sub.parquet")


def test_effective_parallelism_defaults_to_four_when_possible(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("trainer.src.pipelines.predict.os.cpu_count", lambda: 64)

    assert _effective_parallelism(None, 8) == 4


def test_effective_parallelism_caps_at_input_count(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("trainer.src.pipelines.predict.os.cpu_count", lambda: 64)

    assert _effective_parallelism(16, 2) == 2


def test_effective_parallelism_returns_one_for_single_input(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("trainer.src.pipelines.predict.os.cpu_count", lambda: 64)

    assert _effective_parallelism(8, 1) == 1
