from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl

from src.utils.etf_universe import _extract_code_token
from src.utils.meta_sector_map import reload_mapping, sub_to_meta
from src.utils.news_loader import load_raw_news_df


def test_runtime_utils_modules_resolve_local_files() -> None:
    runtime_utils_root = Path(__file__).resolve().parents[1] / "src" / "utils"
    runtime_root = Path(__file__).resolve().parents[1]
    script = """
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import src.utils.meta_sector_map as meta_sector_map_module
import src.utils.news_loader as news_loader_module
import src.utils.etf_universe as etf_universe_module
print(json.dumps({
    "meta_sector_map": meta_sector_map_module.__file__,
    "news_loader": news_loader_module.__file__,
    "etf_universe": etf_universe_module.__file__,
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(runtime_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)

    assert Path(payload["meta_sector_map"]).resolve() == runtime_utils_root / "meta_sector_map.py"
    assert Path(payload["news_loader"]).resolve() == runtime_utils_root / "news_loader.py"
    assert Path(payload["etf_universe"]).resolve() == runtime_utils_root / "etf_universe.py"


def test_runtime_news_loader_falls_back_to_split_parts(tmp_path: Path) -> None:
    merged = tmp_path / "news_merged.parquet"
    part1 = tmp_path / "news_part1.parquet"
    part2 = tmp_path / "news_part2.parquet"

    pl.DataFrame({"title": ["a"], "content": ["x"]}).write_parquet(part1)
    pl.DataFrame({"title": ["b"], "content": ["y"]}).write_parquet(part2)

    df = load_raw_news_df(merged)

    assert df.height == 2


def test_runtime_meta_sector_map_can_reload_custom_mapping(tmp_path: Path) -> None:
    mapping_path = tmp_path / "meta_sector_mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "meta_sectors": {
                    "科技成长": {
                        "sub_categories": ["半导体/芯片"],
                        "market_cap_weight": 1.0,
                        "description": "desc",
                    }
                },
                "global_leader_map": {},
                "notes": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    reload_mapping(mapping_path)

    assert sub_to_meta("半导体/芯片") == "科技成长"


def test_runtime_etf_universe_helpers_still_work() -> None:
    assert _extract_code_token("512480.SH 半导体ETF") == "512480.SH"
    assert _extract_code_token("159995.SZ, 芯片ETF") == "159995.SZ"
