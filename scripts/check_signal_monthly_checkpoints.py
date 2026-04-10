"""Validate and optionally rebuild signals raw parquet from monthly sub inference checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.src.datasets.signals import (  # noqa: E402
    rebuild_signal_raw_from_monthly_checkpoints,
    resolve_signal_monthly_checkpoint_dir,
    validate_signal_monthly_checkpoints,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate signals monthly raw checkpoints.")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("trainer/data/labeled/signals/raw.parquet"),
        help="Target merged raw parquet path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override monthly checkpoint directory.",
    )
    parser.add_argument(
        "--rebuild-raw",
        action="store_true",
        help="Rebuild raw parquet from validated monthly checkpoints.",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or resolve_signal_monthly_checkpoint_dir(args.raw_path)
    month_paths = validate_signal_monthly_checkpoints(args.raw_path, checkpoint_dir=checkpoint_dir)
    print(f"checkpoint_dir={checkpoint_dir}")
    print(f"files={len(month_paths)}")
    print(f"range={month_paths[0].stem} -> {month_paths[-1].stem}")

    if args.rebuild_raw:
        output = rebuild_signal_raw_from_monthly_checkpoints(args.raw_path, checkpoint_dir=checkpoint_dir)
        print(f"rebuilt_raw={output}")


if __name__ == "__main__":
    main()
