from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.io.snapshotting import build_and_write_snapshot


def _load_records(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        return pd.read_parquet(input_path)
    if input_path.suffix.lower() == ".json":
        return pd.read_json(input_path)

    raise ValueError("Unsupported input format. Use .parquet or .json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build snapshot and delta artifacts.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/scholarships_normalized.parquet"),
        help="Input normalized scholarship records (.parquet or .json).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for output artifacts.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Run date in YYYYMMDD format. Defaults to current UTC date.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = _load_records(args.input)
    snapshot_path, changes_path, delta = build_and_write_snapshot(
        records,
        processed_dir=args.processed_dir,
        run_date=args.date,
    )

    print(f"Wrote snapshot: {snapshot_path}")
    print(f"Wrote changes: {changes_path}")
    print(
        "Delta counts: "
        f"added={len(delta['added'])}, "
        f"removed={len(delta['removed'])}, "
        f"changed={len(delta['changed'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

