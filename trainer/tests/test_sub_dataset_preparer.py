import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from trainer.src.datasets.sub import SetFitDatasetPreparer


def _make_preparer(tmp_path: Path) -> SetFitDatasetPreparer:
    preparer = object.__new__(SetFitDatasetPreparer)
    preparer.cfg = SimpleNamespace(seed=42)
    preparer.dcfg = SimpleNamespace(
        raw_data_dir=tmp_path,
        cluster_sampling_majors=["科技信息"],
        random=SimpleNamespace(samples_per_class=2, min_samples_per_class=1),
        cluster=SimpleNamespace(
            n_cap=1000,
            n_clusters=7,
            samples_per_cluster=3,
            min_samples_per_class=2,
            hard_negative_boost=0,
            confused_pairs=[],
        ),
    )
    preparer._embed_model = None
    preparer._embed_lock = None
    return preparer


def test_random_sample_reads_nested_random_config(tmp_path):
    preparer = _make_preparer(tmp_path)
    df = pl.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "label_text": ["x", "x", "y", "y"],
            "major_category": ["科技信息"] * 4,
        }
    )

    sampled = preparer._random_sample(df)

    assert len(sampled) == 4
    counts = sampled.group_by("label_text").len().sort("label_text")
    assert counts["len"].to_list() == [2, 2]


def test_prepare_one_writes_meta_using_dcfg_values(tmp_path, monkeypatch):
    preparer = _make_preparer(tmp_path)
    df = pl.DataFrame(
        {
            "text": ["t1", "t2"],
            "label_text": ["l1", "l2"],
            "major_category": ["科技信息", "科技信息"],
        }
    )

    monkeypatch.setattr(preparer, "_load_raw", lambda major: df)
    monkeypatch.setattr(preparer, "_cluster_sample", lambda in_df: in_df)
    monkeypatch.setattr(preparer, "_random_sample", lambda in_df: in_df)
    dummy_logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
    monkeypatch.setattr("trainer.src.utils.get_logger", lambda: dummy_logger)

    preparer._prepare_one("科技信息")

    cache_dir = tmp_path / "科技信息"
    meta_path = cache_dir / "meta.json"
    prepared_path = cache_dir / "prepared.parquet"

    assert prepared_path.exists()
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["cluster"] is True
    assert meta["n_clusters"] == 7
    assert meta["samples_per_cluster"] == 3
    assert meta["min_samples_per_class"] == 2
