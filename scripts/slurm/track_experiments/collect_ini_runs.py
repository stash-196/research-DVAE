#!/usr/bin/env python3
"""Collect .ini files from experiment dates and report run status.

Scans the given saved_model root for provided dates, parses all .ini files
found under each date, checks whether a "final" .pt file exists in the
same directory, and writes a tabular summary (CSV) where each .ini entry
is a column. Columns are created dynamically to accommodate differing
.ini keys across runs.

Example:
  python3 scripts/collect_ini_runs.py --dates 2026-01-17 2026-01-18 \
    --saved-model-root /flash/DoyaU/stash/research-DVAE/saved_model \
    --output out.csv
"""
import os
import glob
import argparse
import configparser
import yaml
from collections import OrderedDict
from datetime import datetime


def parse_ini(ini_path):
    cp = configparser.ConfigParser()
    try:
        cp.read(ini_path)
    except Exception:
        return {}

    out = {}
    # Include a flattened key: section.option
    for section in cp.sections():
        for k, v in cp[section].items():
            out[f"{section}.{k}"] = v

    # Also include any DEFAULT keys (if present)
    if cp.defaults():
        for k, v in cp.defaults().items():
            out[f"DEFAULT.{k}"] = v

    return out


def find_final_pt(parent_dir):
    # Look for common "final" naming patterns in the same directory
    patterns = ["*final*.pt", "*_final.pt", "*final.pt"]
    for p in patterns:
        matches = glob.glob(os.path.join(parent_dir, p))
        if matches:
            return True, matches
    return False, []


def write_csv(rows, out_file):
    # Dynamically compute union of all keys, preserve order: date, experiment, status, ini_file first
    if not rows:
        print("No rows to write.")
        return

    # Determine all columns
    cols = ["date", "experiment", "status", "ini_file"]
    extra = []
    seen = set(cols)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                extra.append(k)
                seen.add(k)

    cols.extend(sorted(extra))

    # Write CSV
    import csv

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            # ensure present keys
            row = {c: r.get(c, "") for c in cols}
            writer.writerow(row)

    print(f"Wrote summary to {out_file}")


def collect_for_dates(dates, saved_model_root):
    rows = []
    for date in dates:
        print(f"[info] Processing date folder: {date} ...")
        base = os.path.join(saved_model_root, date)
        if not os.path.exists(base):
            print(f"[warn] date folder not found: {base}")
            continue

        # Walk recursively under the date folder and find .ini files
        ini_count = 0
        for root, dirs, files in os.walk(base):
            for fn in files:
                if fn.lower().endswith(".ini"):
                    ini_count += 1

                    # Print progress every 100 files to avoid spamming the log
                    if ini_count % 100 == 0:
                        print(f"       ... processed {ini_count} .ini files inside {date}")

                    ini_path = os.path.join(root, fn)
                    parent_dir = os.path.dirname(ini_path)
                    experiment = os.path.basename(parent_dir)

                    completed, matches = find_final_pt(parent_dir)
                    status = "completed" if completed else "failed"

                    ini_entries = parse_ini(ini_path)

                    row = OrderedDict()
                    row["date"] = date
                    row["experiment"] = experiment
                    row["status"] = status
                    row["ini_file"] = os.path.relpath(ini_path, start=saved_model_root)
                    # add parsed ini entries
                    for k, v in ini_entries.items():
                        row[k] = v

                    # also add found pt file paths (if any)
                    if matches:
                        row["final_pt_paths"] = ";".join(os.path.relpath(m, start=saved_model_root) for m in matches)

                    # Check for evaluation_summary.yaml
                    yaml_path = os.path.join(parent_dir, "evaluation_summary.yaml")
                    if os.path.exists(yaml_path):
                        try:
                            with open(yaml_path, 'r') as yf:
                                yaml_data = yaml.safe_load(yf)
                                if isinstance(yaml_data, dict):
                                    def flatten_dict(d, parent_key='', sep='.'):
                                        items = []
                                        for k, v in d.items():
                                            new_key = f"{parent_key}{sep}{k}" if parent_key else k
                                            if isinstance(v, dict):
                                                items.extend(flatten_dict(v, new_key, sep=sep).items())
                                            else:
                                                items.append((new_key, v))
                                        return dict(items)

                                    flat_yaml = flatten_dict(yaml_data)
                                    for yk, yv in flat_yaml.items():
                                        # Ignore the duplicated .ini configurations often stored under 'config' in yaml
                                        if yk.startswith('config.'):
                                            continue
                                        # Ignore vector/list entries
                                        if isinstance(yv, list):
                                            continue
                                        if yk not in row:
                                            row[yk] = yv
                        except Exception as e:
                            print(f"[warning] Error reading YAML {yaml_path}: {e}")

                    rows.append(row)

        print(f"[info] Finished {date}: found {ini_count} .ini files.")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect .ini files and check final .pt status")
    parser.add_argument("--dates", nargs="*", help="List of date folders (YYYY-MM-DD)")
    parser.add_argument("--dates-file", type=str, help="File with one date per line")
    parser.add_argument("--saved-model-root", type=str, default="/flash/DoyaU/stash/research-DVAE/saved_model", help="Root folder containing saved_model/<date>/...")
    parser.add_argument("--output", type=str, default="experiment_runs_summary.csv", help="Output CSV file")
    args = parser.parse_args()

    dates = []
    if args.dates:
        dates.extend(args.dates)
    if args.dates_file:
        if os.path.exists(args.dates_file):
            with open(args.dates_file) as f:
                for line in f:
                    s = line.strip()
                    if s:
                        dates.append(s)
        else:
            print(f"Dates file not found: {args.dates_file}")

    if not dates:
        print("No dates provided. Use --dates or --dates-file.")
        return

    rows = collect_for_dates(dates, args.saved_model_root)

    if not rows:
        print("No experiments found for given dates.")
        return

    write_csv(rows, args.output)

    # Also attempt to print a quick pandas DataFrame preview if pandas available
    try:
        import pandas as pd

        df = pd.read_csv(args.output)
        print(df.head(20).to_string(index=False))
    except Exception:
        print("Wrote CSV; install pandas to get a prettier preview.")


if __name__ == "__main__":
    main()
