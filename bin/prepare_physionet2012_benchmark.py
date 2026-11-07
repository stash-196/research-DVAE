#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bin Challenge 2012 records for the imputation benchmark (NaNs preserved).

Does not modify the 5-vital ``hourly_v1`` parquet written by
``bin/prepare_physionet2012.py``. This script writes a separate label::

    {data_dir}/physionet2012/processed/hourly_benchmark_v1/stays.parquet

Columns are the 37 SAITS time-series variables plus ``in_hospital_death``
when an ``Outcomes*.txt`` file is present (set-a). Set-b rows stay NaN
on that column.

    python bin/prepare_physionet2012_benchmark.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dvae.dataset.physionet2012_benchmark import (
    BENCHMARK_LABEL,
    MORTALITY_COLUMN,
    records_to_benchmark_frame,
)
from dvae.dataset.physionet2012_dataset import processed_parquet_path
from dvae.utils import find_project_root, load_device_paths


def main(argv: list[str] | None = None) -> int:
    default_data = None
    try:
        root = find_project_root(__file__)
        device = load_device_paths(str(Path(root) / "config" / "device_paths.yaml"))
        default_data = device.get("data_dir") if isinstance(device, dict) else None
    except (FileNotFoundError, ValueError):
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(default_data) if default_data else None,
    )
    parser.add_argument("--label", default=BENCHMARK_LABEL)
    args = parser.parse_args(argv)

    if args.data_dir is None:
        print(
            "Error: --data-dir is required (device_paths.yaml has no data_dir).",
            file=sys.stderr,
        )
        return 1
    raw_dir = args.raw_dir or (Path(args.data_dir) / "physionet2012" / "raw")
    if not raw_dir.is_dir():
        print(
            f"Error: raw dir does not exist: {raw_dir}\n"
            "Download Challenge 2012 set-a/set-b from\n"
            "https://physionet.org/content/challenge-2012/1.0.0/",
            file=sys.stderr,
        )
        return 1
    try:
        frame = records_to_benchmark_frame(raw_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out = processed_parquet_path(args.data_dir, args.label)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    value_cols = [c for c in frame.columns if c not in ("stay_id", "hour", MORTALITY_COLUMN)]
    n_nan = int(frame[value_cols].isna().sum().sum())
    n_vals = int(frame[value_cols].size)
    n_labeled = int(frame.drop_duplicates("stay_id")[MORTALITY_COLUMN].notna().sum())
    print(f"Wrote {out}")
    print(
        f"stays={frame['stay_id'].nunique()} hours={len(frame)} "
        f"nan_frac={n_nan / max(n_vals, 1):.3f} labeled_stays={n_labeled}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
