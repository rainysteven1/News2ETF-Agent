"""Merge raw news parquet files and split them chronologically.

This script is intended for the hopper pipeline:
1. Merge raw news parquet files.
2. Sort by ``datetime`` ascending.
3. Split into train / val / test by chronological order.

The split is ratio-based, but never shuffled, so it remains time-safe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge raw news parquet files and split them by time order.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=[
            "data/converted/tushare_news_2021_today_part1.parquet",
            "data/converted/tushare_news_2021_today_part2.parquet",
        ],
        help="Input parquet files containing raw news.",
    )
    parser.add_argument(
        "--merged-output",
        default="data/converted/tushare_news_2021_today_merged.parquet",
        help="Output parquet path for the merged and sorted news.",
    )
    parser.add_argument(
        "--train-output",
        default="data/converted/tushare_news_train.parquet",
        help="Output parquet path for the train split.",
    )
    parser.add_argument(
        "--val-output",
        default="data/converted/tushare_news_val.parquet",
        help="Output parquet path for the validation split.",
    )
    parser.add_argument(
        "--test-output",
        default="data/converted/tushare_news_test.parquet",
        help="Output parquet path for the test split.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_and_merge(paths: list[Path]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for path in paths:
        df = pl.read_parquet(path)
        required = {"datetime", "content", "title", "source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        frames.append(df.select(["datetime", "content", "title", "source"]))

    merged = pl.concat(frames, how="vertical")
    merged = merged.with_columns(
        pl.col("datetime").str.to_datetime(strict=False).alias("dt")
    ).sort("dt")
    return merged


def split_by_ratio(
    df: pl.DataFrame, train_ratio: float, val_ratio: float
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.slice(0, train_end)
    val_df = df.slice(train_end, val_end - train_end)
    test_df = df.slice(val_end, n - val_end)
    return train_df, val_df, test_df


def summarize_split(name: str, df: pl.DataFrame) -> str:
    if df.is_empty():
        return f"{name}: rows=0"

    start_dt = df.select(pl.col("dt").min()).item()
    end_dt = df.select(pl.col("dt").max()).item()
    return f"{name}: rows={len(df):,}, range={start_dt} -> {end_dt}"


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    input_paths = [Path(p) for p in args.input]
    merged_output = Path(args.merged_output)
    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    test_output = Path(args.test_output)

    merged_df = load_and_merge(input_paths)
    train_df, val_df, test_df = split_by_ratio(merged_df, args.train_ratio, args.val_ratio)

    for path in [merged_output, train_output, val_output, test_output]:
        ensure_parent(path)

    merged_df.drop("dt").write_parquet(merged_output)
    train_df.drop("dt").write_parquet(train_output)
    val_df.drop("dt").write_parquet(val_output)
    test_df.drop("dt").write_parquet(test_output)

    print(f"merged: rows={len(merged_df):,}")
    print(summarize_split("train", train_df))
    print(summarize_split("val", val_df))
    print(summarize_split("test", test_df))
    print(f"merged output: {merged_output}")
    print(f"train output: {train_output}")
    print(f"val output: {val_output}")
    print(f"test output: {test_output}")


if __name__ == "__main__":
    main()
