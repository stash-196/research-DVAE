#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bin PhysioNet Challenge 2012 records onto an hourly grid (NaNs preserved).

``data_dir`` comes from config/device_paths.yaml for this hostname:
  Studio  ~/mounts/bucket/DoyaU/stash/research-DVAE/data
  Deigo   /bucket/DoyaU/stash/research-DVAE/data

Raw records default to {data_dir}/physionet2012/raw (set-a/, set-b/).
On Studio that is the bucket tree via ~/mounts; do not pass /bucket/... here.

    python bin/prepare_physionet2012.py

Writes {data_dir}/physionet2012/processed/hourly_v1/stays.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dvae.dataset.physionet2012_dataset import (
    DEFAULT_LABEL,
    processed_parquet_path,
    records_to_hourly_frame,
)
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
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing set-a/ and/or set-b/ (or loose .txt records).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(default_data) if default_data else None,
        help="Project data_dir (processed files go under physionet2012/processed/).",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"Processed corpus name (default {DEFAULT_LABEL}).",
    )
    args = parser.parse_args(argv)

    if args.data_dir is None:
        print("Error: --data-dir is required (device_paths.yaml has no data_dir).", file=sys.stderr)
        return 1
    raw_dir = args.raw_dir
    if raw_dir is None:
        raw_dir = Path(args.data_dir) / "physionet2012" / "raw"
    if not raw_dir.is_dir():
        print(
            f"Error: raw dir does not exist: {raw_dir}\n"
            "Download Challenge 2012 set-a/set-b from\n"
            "https://physionet.org/content/challenge-2012/1.0.0/",
            file=sys.stderr,
        )
        return 1

    try:
        frame = records_to_hourly_frame(raw_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out = processed_parquet_path(args.data_dir, args.label)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    n_stays = frame["stay_id"].nunique()
    n_nan = int(frame.drop(columns=["stay_id", "hour"]).isna().sum().sum())
    n_vals = int(frame.drop(columns=["stay_id", "hour"]).size)
    print(f"Wrote {out}")
    print(f"stays={n_stays} hours={len(frame)} nan_frac={n_nan / max(n_vals, 1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
