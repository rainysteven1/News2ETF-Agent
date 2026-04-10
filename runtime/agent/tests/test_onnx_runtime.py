from __future__ import annotations

import json

from src.signals import label_stats, onnx_inference


def test_onnx_inference_uses_runtime_label_stats() -> None:
    assert onnx_inference.LabelStats is label_stats.LabelStats
    assert onnx_inference.safe_name is label_stats.safe_name


def test_runtime_label_stats_loads_from_explicit_path(tmp_path) -> None:
    stats_path = tmp_path / "label_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "major_category": {"科技信息": 2, "医药健康": 1},
                "sub_category_by_major": {
                    "科技信息": {"半导体/芯片": 2},
                    "医药健康": {"生物医药/创新药": 1},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded = label_stats.LabelStats.load(stats_path)

    assert loaded.get_major_categories() == ["医药健康", "科技信息"]
    assert loaded.get_sub_categories("科技信息") == ["半导体/芯片"]
    assert label_stats.safe_name("半导体/芯片") == "半导体_芯片"
