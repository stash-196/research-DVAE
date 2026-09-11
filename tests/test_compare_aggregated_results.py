"""Unit tests for overlaying finished aggregate CSVs."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from dvae.eval.compare_aggregated_results import (
    ExperimentAggregate,
    build_combined_rows,
    intersect_x_values,
    load_experiments,
    main,
    parse_experiment_spec,
    resolve_aggregated_csv,
    run_compare,
    summarize_metric_by_x,
)


def _write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_parse_experiment_spec_pipe_and_equals():
    assert parse_experiment_spec("OTF|/tmp/exp") == ("OTF", "/tmp/exp")
    assert parse_experiment_spec("interpolate=/tmp/other") == (
        "interpolate",
        "/tmp/other",
    )
    with pytest.raises(ValueError, match="label\\|path"):
        parse_experiment_spec("no-separator")


def test_resolve_csv_from_file_dir_and_exp_root(tmp_path: Path):
    csv_path = (
        tmp_path
        / "exp"
        / "aggregate_eval_plots_sampling_ratio"
        / "aggregated_values.csv"
    )
    _write_csv(
        csv_path,
        [{"sampling_ratio": "0.1", "kld_auto": "1.0"}],
    )

    assert resolve_aggregated_csv(str(csv_path)) == str(csv_path.resolve())
    assert resolve_aggregated_csv(str(csv_path.parent)) == str(csv_path.resolve())
    assert resolve_aggregated_csv(str(tmp_path / "exp")) == str(csv_path.resolve())


def test_resolve_csv_prefers_x_parameter_dir(tmp_path: Path):
    ratio_csv = (
        tmp_path / "aggregate_eval_plots_sampling_ratio" / "aggregated_values.csv"
    )
    other_csv = tmp_path / "aggregate_eval_plots_mask_label" / "aggregated_values.csv"
    _write_csv(ratio_csv, [{"sampling_ratio": "0.1", "kld_auto": "1.0"}])
    _write_csv(other_csv, [{"mask_label": "0.2", "kld_auto": "2.0"}])

    resolved = resolve_aggregated_csv(str(tmp_path), x_parameter="sampling_ratio")
    assert resolved == str(ratio_csv.resolve())


def test_resolve_csv_missing_fails_clearly(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="aggregated_values.csv"):
        resolve_aggregated_csv(str(tmp_path / "missing_exp"))


def test_resolve_csv_ambiguous_dirs_fail(tmp_path: Path):
    _write_csv(
        tmp_path / "aggregate_eval_plots_foo" / "aggregated_values.csv",
        [{"sampling_ratio": "0.1", "kld_auto": "1.0"}],
    )
    _write_csv(
        tmp_path / "aggregate_eval_plots_bar" / "aggregated_values.csv",
        [{"sampling_ratio": "0.1", "kld_auto": "2.0"}],
    )
    with pytest.raises(FileNotFoundError, match="Multiple"):
        resolve_aggregated_csv(str(tmp_path), x_parameter="sampling_ratio")


def test_summarize_mean_std_from_replicates():
    rows = [
        {"sampling_ratio": "0.5", "kld_auto": "1.0"},
        {"sampling_ratio": "0.5", "kld_auto": "3.0"},
        {"sampling_ratio": "0.8", "kld_auto": "4.0"},
    ]
    series = summarize_metric_by_x(rows, "sampling_ratio", "kld_auto")
    by_x = {point["x"]: point for point in series}
    assert by_x[0.5]["mean"] == pytest.approx(2.0)
    assert by_x[0.5]["std"] == pytest.approx(2.0**0.5)
    assert by_x[0.5]["n"] == 2
    assert by_x[0.8]["mean"] == pytest.approx(4.0)
    assert by_x[0.8]["std"] is None


def test_summarize_uses_std_column_when_single_row():
    rows = [
        {
            "sampling_ratio": "0.2",
            "kld_auto": "1.5",
            "kld_auto_std_across_batches": "0.25",
        }
    ]
    series = summarize_metric_by_x(rows, "sampling_ratio", "kld_auto")
    assert series[0]["std"] == pytest.approx(0.25)


def test_intersect_x_values_warns_and_keeps_common(capsys):
    series_by_label = {
        "OTF": [{"x": 0.1, "mean": 1.0, "std": None, "n": 1}],
        "interpolate": [
            {"x": 0.1, "mean": 2.0, "std": None, "n": 1},
            {"x": 0.9, "mean": 3.0, "std": None, "n": 1},
        ],
    }
    common = intersect_x_values(series_by_label)
    assert common == [0.1]
    captured = capsys.readouterr().out
    assert "grids differ" in captured
    assert "0.9" in captured


def test_cli_two_synthetic_csvs_writes_pngs_and_combined(tmp_path: Path):
    otf_rows = [
        {"sampling_ratio": "0.0", "kld_auto": "10.0", "spectrum_error_auto": "0.40"},
        {"sampling_ratio": "0.4", "kld_auto": "4.0", "spectrum_error_auto": "0.20"},
        {"sampling_ratio": "0.7", "kld_auto": "1.5", "spectrum_error_auto": "0.10"},
    ]
    interp_rows = [
        {"sampling_ratio": "0.0", "kld_auto": "12.0", "spectrum_error_auto": "0.50"},
        {"sampling_ratio": "0.4", "kld_auto": "6.0", "spectrum_error_auto": "0.30"},
        {"sampling_ratio": "0.7", "kld_auto": "2.0", "spectrum_error_auto": "0.15"},
        {"sampling_ratio": "0.8", "kld_auto": "1.0", "spectrum_error_auto": "0.12"},
    ]
    otf_csv = (
        tmp_path
        / "otf"
        / "aggregate_eval_plots_sampling_ratio"
        / "aggregated_values.csv"
    )
    interp_csv = (
        tmp_path
        / "interpolate"
        / "aggregate_eval_plots_sampling_ratio"
        / "aggregated_values.csv"
    )
    _write_csv(otf_csv, otf_rows)
    _write_csv(interp_csv, interp_rows)
    out_dir = tmp_path / "compare_out"

    result = run_compare(
        experiment_specs=[
            f"OTF|{tmp_path / 'otf'}",
            f"interpolate|{interp_csv.parent}",
        ],
        metrics=["kld_auto", "spectrum_error_auto"],
        x_parameter="sampling_ratio",
        output_dir=str(out_dir),
    )

    kld_png = out_dir / "compare_kld_auto_vs_sampling_ratio.png"
    spec_png = out_dir / "compare_spectrum_error_auto_vs_sampling_ratio.png"
    combined = out_dir / "compare_aggregated_values.csv"
    assert kld_png.is_file() and kld_png.stat().st_size > 0
    assert spec_png.is_file() and spec_png.stat().st_size > 0
    assert combined.is_file()
    assert result["combined_csv"] == str(combined)
    assert {os.path.basename(p) for p in result["plot_paths"]} == {
        kld_png.name,
        spec_png.name,
    }

    with combined.open() as handle:
        rows = list(csv.DictReader(handle))
    labels = {row["experiment"] for row in rows}
    assert labels == {"OTF", "interpolate"}
    otf_ratios = [
        float(row["sampling_ratio"]) for row in rows if row["experiment"] == "OTF"
    ]
    interp_ratios = [
        float(row["sampling_ratio"])
        for row in rows
        if row["experiment"] == "interpolate"
    ]
    assert otf_ratios == [0.0, 0.4, 0.7]
    assert interp_ratios == [0.0, 0.4, 0.7, 0.8]

    cli_out = tmp_path / "compare_cli"
    assert (
        main(
            [
                "--experiments",
                f"OTF|{tmp_path / 'otf'}",
                f"interpolate={interp_csv}",
                "--metrics",
                "kld_auto",
                "spectrum_error_auto",
                "--x-parameter",
                "sampling_ratio",
                "--output_dir",
                str(cli_out),
            ]
        )
        == 0
    )
    assert (cli_out / "compare_kld_auto_vs_sampling_ratio.png").is_file()
    assert (cli_out / "compare_aggregated_values.csv").is_file()


def test_missing_csv_fails_clearly(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="aggregated_values.csv"):
        load_experiments([f"OTF|{tmp_path / 'nope'}"], x_parameter="sampling_ratio")


def test_main_cli_missing_csv_returns_1(tmp_path: Path):
    assert (
        main(
            [
                "--experiments",
                f"OTF|{tmp_path / 'missing'}",
                "--output_dir",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )


def test_build_combined_rows_includes_std():

    otf = ExperimentAggregate(
        "OTF",
        "/tmp/otf.csv",
        [{"sampling_ratio": "0.1", "kld_auto": "1.0"}],
    )
    summaries = {
        "OTF": {
            "kld_auto": [{"x": 0.1, "mean": 1.0, "std": 0.2, "n": 2}],
        }
    }
    rows = build_combined_rows([otf], ["kld_auto"], "sampling_ratio", summaries)
    assert rows[0]["kld_auto"] == 1.0
    assert rows[0]["kld_auto_std"] == 0.2
    assert rows[0]["n"] == 2
