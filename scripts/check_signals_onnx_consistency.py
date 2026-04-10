"""Compare signals ONNX inference outputs against a reference feature parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent / "runtime" / "agent"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _numeric_overlap_columns(df_left: pl.DataFrame, df_right: pl.DataFrame) -> list[str]:
    common = set(df_left.columns) & set(df_right.columns)
    excluded_prefixes = ("lgbm_score_",)
    excluded_exact = {"date"}
    cols: list[str] = []
    for col in sorted(common):
        if col in excluded_exact or any(col.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if df_left.schema[col].is_numeric() and df_right.schema[col].is_numeric():
            cols.append(col)
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Check consistency between signals ONNX inference and reference features.")
    parser.add_argument("--bundle-dir", type=Path, default=Path("trainer/models/signals/latest"))
    parser.add_argument("--reference", type=Path, default=Path("data/agent_features.parquet"))
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--limit-dates", type=int, default=30)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    from src.signals.signals_inference import SignalsONNXInferencePipeline
    from trainer.src.config import load_config
    from trainer.src.datasets.signals import WeeklySignalDataset

    cfg = load_config().signals
    dataset = WeeklySignalDataset(cfg.dataset, force=False, ohlcv_cfg=cfg.ohlcv)
    assert dataset.sentiment_df is not None, "Signals sentiment dataset unavailable"
    sentiment_df = dataset.sentiment_df

    if not args.reference.exists():
        raise FileNotFoundError(f"Reference feature parquet not found: {args.reference}")

    pipeline = SignalsONNXInferencePipeline(
        bundle_dir=args.bundle_dir,
        meta_sector_mapping_path=Path("data/meta_sector_mapping.json"),
    )
    inferred = pipeline.infer_feature_frame(
        sentiment_df,
        start_date=args.start_date,
        end_date=args.end_date,
    ).with_columns(pl.col("date").cast(pl.Utf8))
    reference = pl.read_parquet(args.reference).with_columns(pl.col("date").cast(pl.Utf8))

    overlap_dates = sorted(set(inferred["date"].to_list()) & set(reference["date"].to_list()))
    if args.limit_dates and len(overlap_dates) > args.limit_dates:
        overlap_dates = overlap_dates[-args.limit_dates :]
    if not overlap_dates:
        raise ValueError("No overlapping dates between ONNX inference and reference parquet.")

    inferred = inferred.filter(pl.col("date").is_in(overlap_dates)).sort("date")
    reference = reference.filter(pl.col("date").is_in(overlap_dates)).sort("date")
    joined = inferred.join(reference, on="date", how="inner", suffix="_ref")
    compare_cols = _numeric_overlap_columns(inferred, reference)
    if not compare_cols:
        raise ValueError("No comparable numeric columns found.")

    max_abs_diff = 0.0
    bad_cols: list[tuple[str, float]] = []
    for col in compare_cols:
        lhs = joined[col].to_numpy().astype(np.float64)
        rhs = joined[f"{col}_ref"].to_numpy().astype(np.float64)
        diff = float(np.max(np.abs(lhs - rhs))) if len(lhs) else 0.0
        max_abs_diff = max(max_abs_diff, diff)
        if diff > args.atol:
            bad_cols.append((col, diff))

    print(f"checked_dates={len(overlap_dates)}")
    print(f"compared_columns={len(compare_cols)}")
    print(f"max_abs_diff={max_abs_diff:.8f}")
    if bad_cols:
        print("mismatched_columns:")
        for col, diff in sorted(bad_cols, key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {col}: {diff:.8f}")
        raise SystemExit(1)
    print("consistency_check=PASS")


if __name__ == "__main__":
    main()
