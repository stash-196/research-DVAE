"""Unit tests for overlaying finished aggregate CSVs."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

import dvae.eval.compare_aggregated_results as compare_mod
from dvae.eval.aggregate_plot_style import (
    DEFAULT_COMPARE_METRICS,
    choose_y_axis_scale,
    error_band_edges,
    get_metric_display_name,
    save_figure,
)
from dvae.eval.compare_aggregated_results import (
    ExperimentAggregate,
    build_arg_parser,
    build_combined_rows,
    expand_rows_to_channel_long,
    intersect_x_values,
    load_experiments,
    main,
    normalize_channel_label,
    parse_experiment_spec,
    plot_metric_channel_panels,
    plot_metric_overlay,
    resolve_aggregated_csv,
    run_compare,
    summarize_metric_by_x,
    summarize_metric_by_x_channel,
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
    assert "channel" not in rows[0]
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


def test_normalize_channel_label_maps_raw_ch_and_suffixes():
    assert normalize_channel_label("raw_ch1") == "ch1"
    assert normalize_channel_label("raw_ch2_interpolate") == "ch2"
    assert normalize_channel_label("raw_ch3_indicate") == "ch3"
    assert normalize_channel_label("ch4") == "ch4"
    assert normalize_channel_label("Ch1") == "ch1"
    assert normalize_channel_label("raw_all") is None
    assert normalize_channel_label("") is None


def _one_d_sep_rows():
    """Long-form 1d-sep aggregate: one row per (ratio, observation_process)."""
    rows = []
    for ratio, scores in (
        ("0.0", (10.0, 20.0, 30.0, 40.0)),
        ("0.4", (4.0, 8.0, 12.0, 16.0)),
        ("0.7", (1.0, 2.0, 3.0, 4.0)),
    ):
        for idx, score in enumerate(scores, start=1):
            rows.append(
                {
                    "sampling_ratio": ratio,
                    "observation_process": f"raw_ch{idx}",
                    "kld_auto": str(score),
                    "spectrum_error_auto": str(score / 100.0),
                }
            )
    return rows


def _joint_4d_rows():
    """Wide-form joint-4d aggregate: one row per sampling_ratio."""
    return [
        {
            "sampling_ratio": "0.0",
            "kld_auto": "25.0",
            "kld_auto_ch1": "11.0",
            "kld_auto_ch2": "21.0",
            "kld_auto_ch3": "31.0",
            "kld_auto_ch4": "41.0",
            "spectrum_error_auto": "0.25",
            "spectrum_error_auto_ch1": "0.11",
            "spectrum_error_auto_ch2": "0.21",
            "spectrum_error_auto_ch3": "0.31",
            "spectrum_error_auto_ch4": "0.41",
        },
        {
            "sampling_ratio": "0.4",
            "kld_auto": "10.0",
            "kld_auto_ch1": "5.0",
            "kld_auto_ch2": "9.0",
            "kld_auto_ch3": "13.0",
            "kld_auto_ch4": "17.0",
            "spectrum_error_auto": "0.10",
            "spectrum_error_auto_ch1": "0.05",
            "spectrum_error_auto_ch2": "0.09",
            "spectrum_error_auto_ch3": "0.13",
            "spectrum_error_auto_ch4": "0.17",
        },
        {
            "sampling_ratio": "0.7",
            "kld_auto": "2.5",
            "kld_auto_ch1": "1.5",
            "kld_auto_ch2": "2.5",
            "kld_auto_ch3": "3.5",
            "kld_auto_ch4": "4.5",
            "spectrum_error_auto": "0.025",
            "spectrum_error_auto_ch1": "0.015",
            "spectrum_error_auto_ch2": "0.025",
            "spectrum_error_auto_ch3": "0.035",
            "spectrum_error_auto_ch4": "0.045",
        },
        {
            "sampling_ratio": "0.9",
            "kld_auto": "1.0",
            "kld_auto_ch1": "0.5",
            "kld_auto_ch2": "0.6",
            "kld_auto_ch3": "0.7",
            "kld_auto_ch4": "0.8",
            "spectrum_error_auto": "0.01",
            "spectrum_error_auto_ch1": "0.005",
            "spectrum_error_auto_ch2": "0.006",
            "spectrum_error_auto_ch3": "0.007",
            "spectrum_error_auto_ch4": "0.008",
        },
    ]


def test_expand_long_form_1d_sep_uses_base_metric():
    points = expand_rows_to_channel_long(
        _one_d_sep_rows(), "sampling_ratio", "kld_auto"
    )
    by_key = {(p["x"], p["channel"]): p["value"] for p in points}
    assert by_key[(0.0, "ch1")] == pytest.approx(10.0)
    assert by_key[(0.4, "ch3")] == pytest.approx(12.0)
    assert by_key[(0.7, "ch4")] == pytest.approx(4.0)
    assert len(points) == 12


def test_expand_wide_form_joint_4d_uses_metric_ch_columns():
    points = expand_rows_to_channel_long(
        _joint_4d_rows(), "sampling_ratio", "kld_auto"
    )
    by_key = {(p["x"], p["channel"]): p["value"] for p in points}
    assert by_key[(0.0, "ch1")] == pytest.approx(11.0)
    assert by_key[(0.4, "ch2")] == pytest.approx(9.0)
    assert (0.9, "ch4") in by_key
    # Base-column mean must not be treated as a channel.
    assert all(p["channel"].startswith("ch") for p in points)
    assert len(points) == 16


def test_summarize_channel_long_groups_replicates():
    points = [
        {"x": 0.5, "channel": "ch1", "value": 1.0, "std": None},
        {"x": 0.5, "channel": "ch1", "value": 3.0, "std": None},
        {"x": 0.5, "channel": "ch2", "value": 8.0, "std": 0.4},
    ]
    series = summarize_metric_by_x_channel(points)
    by_key = {(p["x"], p["channel"]): p for p in series}
    assert by_key[(0.5, "ch1")]["mean"] == pytest.approx(2.0)
    assert by_key[(0.5, "ch1")]["std"] == pytest.approx(2.0**0.5)
    assert by_key[(0.5, "ch1")]["n"] == 2
    assert by_key[(0.5, "ch2")]["mean"] == pytest.approx(8.0)
    assert by_key[(0.5, "ch2")]["std"] == pytest.approx(0.4)


def test_default_path_collapses_1d_rows_by_sampling_ratio_only(tmp_path: Path):
    """Unset --channel-pair still mean-collapses extra observation_process rows."""
    one_d_csv = (
        tmp_path
        / "one_d"
        / "aggregate_eval_plots_sampling_ratio"
        / "aggregated_values.csv"
    )
    _write_csv(one_d_csv, _one_d_sep_rows())
    out_dir = tmp_path / "default_out"

    result = run_compare(
        experiment_specs=[f"1d-sep|{tmp_path / 'one_d'}"],
        metrics=["kld_auto"],
        x_parameter="sampling_ratio",
        output_dir=str(out_dir),
    )

    combined = out_dir / "compare_aggregated_values.csv"
    assert combined.is_file()
    assert result["combined_csv"] == str(combined)
    assert (out_dir / "compare_kld_auto_vs_sampling_ratio.png").is_file()
    assert not (out_dir / "compare_kld_auto_vs_sampling_ratio_by_channel.png").exists()

    with combined.open() as handle:
        rows = list(csv.DictReader(handle))
    assert "channel" not in rows[0]
    by_ratio = {float(row["sampling_ratio"]): row for row in rows}
    # raw_ch1..4 at 0.0 are 10, 20, 30, 40 → mean 25, n=4
    assert float(by_ratio[0.0]["kld_auto"]) == pytest.approx(25.0)
    assert int(by_ratio[0.0]["n"]) == 4
    assert float(by_ratio[0.4]["kld_auto"]) == pytest.approx(10.0)
    assert float(by_ratio[0.7]["kld_auto"]) == pytest.approx(2.5)


def test_channel_pair_joins_1d_long_and_4d_wide(tmp_path: Path):
    one_d_csv = (
        tmp_path
        / "one_d"
        / "aggregate_eval_plots_sampling_ratio"
        / "aggregated_values.csv"
    )
    four_d_csv = (
        tmp_path
        / "four_d"
        / "aggregate_eval_plots_sampling_ratio"
        / "aggregated_values.csv"
    )
    _write_csv(one_d_csv, _one_d_sep_rows())
    _write_csv(four_d_csv, _joint_4d_rows())
    out_dir = tmp_path / "channel_pair_out"

    result = run_compare(
        experiment_specs=[
            f"1d-sep|{tmp_path / 'one_d'}",
            f"OTF-4d|{tmp_path / 'four_d'}",
        ],
        metrics=["kld_auto", "spectrum_error_auto"],
        x_parameter="sampling_ratio",
        output_dir=str(out_dir),
        channel_pair=True,
    )

    kld_png = out_dir / "compare_kld_auto_vs_sampling_ratio_by_channel.png"
    spec_png = out_dir / "compare_spectrum_error_auto_vs_sampling_ratio_by_channel.png"
    combined = out_dir / "compare_aggregated_values.csv"
    assert kld_png.is_file() and kld_png.stat().st_size > 0
    assert spec_png.is_file() and spec_png.stat().st_size > 0
    # Default-style single-panel names must not be written in this mode.
    assert not (out_dir / "compare_kld_auto_vs_sampling_ratio.png").exists()
    assert {os.path.basename(p) for p in result["plot_paths"]} == {
        kld_png.name,
        spec_png.name,
    }

    with combined.open() as handle:
        rows = list(csv.DictReader(handle))
    assert "channel" in rows[0]
    labels = {row["experiment"] for row in rows}
    assert labels == {"1d-sep", "OTF-4d"}

    one_d = {
        (float(row["sampling_ratio"]), row["channel"]): float(row["kld_auto"])
        for row in rows
        if row["experiment"] == "1d-sep"
    }
    four_d = {
        (float(row["sampling_ratio"]), row["channel"]): float(row["kld_auto"])
        for row in rows
        if row["experiment"] == "OTF-4d"
    }
    assert one_d[(0.0, "ch1")] == pytest.approx(10.0)
    assert four_d[(0.0, "ch1")] == pytest.approx(11.0)
    assert one_d[(0.4, "ch2")] == pytest.approx(8.0)
    assert four_d[(0.4, "ch2")] == pytest.approx(9.0)
    assert one_d[(0.7, "ch4")] == pytest.approx(4.0)
    assert four_d[(0.7, "ch4")] == pytest.approx(4.5)
    # 4d-only extra ratio is kept in the combined CSV (plots use the intersection).
    assert (0.9, "ch1") in four_d
    assert (0.9, "ch1") not in one_d
    assert {ch for _, ch in one_d} == {"ch1", "ch2", "ch3", "ch4"}

    cli_out = tmp_path / "channel_pair_cli"
    assert (
        main(
            [
                "--experiments",
                f"1d-sep|{tmp_path / 'one_d'}",
                f"OTF-4d|{four_d_csv}",
                "--metrics",
                "kld_auto",
                "--x-parameter",
                "sampling_ratio",
                "--channel-pair",
                "--output_dir",
                str(cli_out),
            ]
        )
        == 0
    )
    assert (cli_out / "compare_kld_auto_vs_sampling_ratio_by_channel.png").is_file()
    with (cli_out / "compare_aggregated_values.csv").open() as handle:
        cli_rows = list(csv.DictReader(handle))
    assert "channel" in cli_rows[0]


def _fill_between_y(ax):
    chunks = []
    for coll in ax.collections:
        for path in coll.get_paths():
            verts = path.vertices
            if len(verts):
                chunks.append(verts[:, 1])
    assert chunks, "expected a mean±std fill_between band"
    return np.concatenate(chunks)


def _capture_saved_axes(monkeypatch):
    """Record y-scale and band vertices from figures passed to ``save_figure``."""
    captured = []

    def _save(fig, path, left_margin=None):
        for ax in fig.axes:
            if not ax.lines:
                continue
            captured.append(
                {
                    "yscale": ax.get_yscale(),
                    "band_y": _fill_between_y(ax),
                }
            )
        save_figure(fig, path, left_margin=left_margin)

    monkeypatch.setattr(compare_mod, "save_figure", _save)
    return captured


def _overlay_experiment():
    return ExperimentAggregate(
        "A",
        "/tmp/a.csv",
        [{"sampling_ratio": "0.1", "kld_auto": "1.0"}],
    )


def test_error_band_edges_stay_true_off_log_and_positive_on_log():
    lo, hi = error_band_edges(2.0, 5.0, log_scale=False)
    assert lo == pytest.approx(-3.0)
    assert lo <= 0
    assert hi == pytest.approx(7.0)

    lo, hi = error_band_edges(2.0, 5.0, log_scale=True)
    assert lo > 0
    assert lo == pytest.approx(max(float(np.nextafter(0.0, 1.0)), 2.0 * 1e-3))
    assert hi == pytest.approx(7.0)

    # Already-positive bands are unchanged on a log axis.
    lo, hi = error_band_edges(5.0, 1.0, log_scale=True)
    assert lo == pytest.approx(4.0)
    assert hi == pytest.approx(6.0)


def test_choose_y_axis_scale_log_only_when_strictly_positive():
    assert choose_y_axis_scale([0.02, 0.019, 50.0, 51.0]) == "log"
    assert choose_y_axis_scale([-150.0, 100.0, 350.0]) == "symlog"
    assert choose_y_axis_scale([-1.0, 1.0, 3.0]) == "linear"
    assert choose_y_axis_scale([]) == "linear"


def test_overlay_std_wider_than_mean_keeps_lower_edge_on_linear(tmp_path, monkeypatch):
    captured = _capture_saved_axes(monkeypatch)
    series = [
        {"x": 0.1, "mean": 1.0, "std": 2.0, "n": 2},
        {"x": 0.5, "mean": 1.5, "std": 0.2, "n": 2},
    ]
    out = tmp_path / "linear.png"
    assert plot_metric_overlay(
        [_overlay_experiment()],
        "kld_auto",
        "sampling_ratio",
        [0.1, 0.5],
        {"A": series},
        str(out),
    )
    assert out.is_file()
    assert captured[0]["yscale"] == "linear"
    assert captured[0]["band_y"].min() == pytest.approx(1.0 - 2.0)
    assert captured[0]["band_y"].min() <= 0


def test_overlay_std_wider_than_mean_keeps_lower_edge_on_symlog(tmp_path, monkeypatch):
    captured = _capture_saved_axes(monkeypatch)
    series = [
        {"x": 0.1, "mean": 100.0, "std": 250.0, "n": 2},
        {"x": 0.5, "mean": 80.0, "std": 10.0, "n": 2},
    ]
    out = tmp_path / "symlog.png"
    assert plot_metric_overlay(
        [_overlay_experiment()],
        "kld_auto",
        "sampling_ratio",
        [0.1, 0.5],
        {"A": series},
        str(out),
    )
    assert captured[0]["yscale"] == "symlog"
    assert captured[0]["band_y"].min() == pytest.approx(100.0 - 250.0)
    assert captured[0]["band_y"].min() <= 0


def test_overlay_log_axis_clamps_nonpositive_lower_edge(tmp_path, monkeypatch):
    """When the chosen scale is log, a std wider than the mean stays above 0."""
    captured = _capture_saved_axes(monkeypatch)
    monkeypatch.setattr(compare_mod, "choose_y_axis_scale", lambda _values: "log")
    series = [
        {"x": 0.1, "mean": 2.0, "std": 5.0, "n": 2},
        {"x": 0.5, "mean": 4.0, "std": 1.0, "n": 2},
    ]
    out = tmp_path / "log_clamp.png"
    assert plot_metric_overlay(
        [_overlay_experiment()],
        "kld_auto",
        "sampling_ratio",
        [0.1, 0.5],
        {"A": series},
        str(out),
    )
    floor = max(float(np.nextafter(0.0, 1.0)), 2.0 * 1e-3)
    assert captured[0]["yscale"] == "log"
    assert captured[0]["band_y"].min() == pytest.approx(floor)
    assert captured[0]["band_y"].min() > 0


def test_overlay_log_axis_keeps_strictly_positive_band(tmp_path, monkeypatch):
    captured = _capture_saved_axes(monkeypatch)
    series = [
        {"x": 0.1, "mean": 0.02, "std": 0.001, "n": 2},
        {"x": 0.5, "mean": 50.0, "std": 1.0, "n": 2},
    ]
    out = tmp_path / "log.png"
    assert plot_metric_overlay(
        [_overlay_experiment()],
        "kld_auto",
        "sampling_ratio",
        [0.1, 0.5],
        {"A": series},
        str(out),
    )
    assert captured[0]["yscale"] == "log"
    assert captured[0]["band_y"].min() == pytest.approx(0.02 - 0.001)
    assert captured[0]["band_y"].min() > 0


def test_channel_panels_symlog_band_reaches_mean_minus_std(tmp_path, monkeypatch):
    captured = _capture_saved_axes(monkeypatch)
    series = [
        {"x": 0.1, "mean": 100.0, "std": 250.0, "n": 2},
        {"x": 0.5, "mean": 80.0, "std": 10.0, "n": 2},
    ]
    out = tmp_path / "panels.png"
    summaries = {"A": {"ch1": series}}
    assert plot_metric_channel_panels(
        [_overlay_experiment()],
        "kld_auto",
        "sampling_ratio",
        summaries,
        str(out),
    )
    drawn = [item for item in captured if item["yscale"] == "symlog"]
    assert len(drawn) == 1
    assert drawn[0]["band_y"].min() == pytest.approx(100.0 - 250.0)
    assert drawn[0]["band_y"].min() <= 0


def _dynamical_compare_rows(scale):
    """sampling_ratio rows with the scalar dynamical scores (no spectrum list)."""
    rows = []
    for ratio, lyap, jac_mean, cross in (
        (0.0, 0.40, 1.6, -0.30),
        (0.4, 0.15, 1.1, -0.05),
        (0.7, -0.05, 0.8, 0.10),
    ):
        rows.append(
            {
                "sampling_ratio": str(ratio),
                "kld_auto": str(10.0 * scale * (1.0 - ratio)),
                "spectrum_error_auto": str(0.4 * scale * (1.0 - 0.5 * ratio)),
                "lyap_max": str(lyap * scale),
                "jac_opnorm_mean": str(jac_mean),
                "jac_opnorm_max": str(jac_mean + 0.4),
                "jac_rho_max": str(jac_mean - 0.2),
                "jac_rho_gt1_frac": str(0.2 + 0.3 * ratio),
                "local_drift_avg_d_norm": str(0.05 * scale * (1.0 + ratio)),
                "local_drift_avg_cross_term": str(cross),
                "local_drift_avg_delta_mse": str(cross + 0.02),
                "id_pr_auto": str(2.0 + ratio),
            }
        )
    return rows


def test_compare_defaults_include_dynamical_scalars():
    defaults = build_arg_parser().get_default("metrics")
    for key in (
        "kld_auto",
        "spectrum_error_auto",
        "lyap_max",
        "jac_opnorm_mean",
        "jac_opnorm_max",
        "jac_rho_max",
        "jac_rho_gt1_frac",
        "local_drift_avg_d_norm",
        "local_drift_avg_cross_term",
        "local_drift_avg_delta_mse",
    ):
        assert key in defaults
    assert defaults == list(DEFAULT_COMPARE_METRICS)
    assert "lyap_spectrum" not in defaults
    assert "id_pr_auto" not in defaults
    assert "id_pr_joint_gt" not in defaults
    for key in (
        "id_pr_hidden_joint_tf",
        "id_pr_hidden_joint_auto",
        "id_twonn_hidden_joint_tf",
        "id_twonn_hidden_joint_auto",
    ):
        assert key in defaults
    help_text = build_arg_parser().format_help()
    assert "id_pr_" in help_text
    assert "lyap_spectrum" in help_text
    assert get_metric_display_name("lyap_max") == "Max Lyapunov exponent"


def test_compare_overlay_writes_dynamical_metric_pngs(tmp_path: Path):
    otf_csv = (
        tmp_path / "otf" / "aggregate_eval_plots_sampling_ratio" / "aggregated_values.csv"
    )
    interp_csv = (
        tmp_path
        / "interpolate"
        / "aggregate_eval_plots_sampling_ratio"
        / "aggregated_values.csv"
    )
    _write_csv(otf_csv, _dynamical_compare_rows(1.0))
    _write_csv(interp_csv, _dynamical_compare_rows(1.4))
    out_dir = tmp_path / "compare_dyn"
    metrics = [
        "lyap_max",
        "jac_opnorm_mean",
        "jac_opnorm_max",
        "local_drift_avg_d_norm",
        "local_drift_avg_cross_term",
        "local_drift_avg_delta_mse",
        "id_pr_auto",
    ]
    result = run_compare(
        experiment_specs=[
            f"OTF|{tmp_path / 'otf'}",
            f"interpolate|{tmp_path / 'interpolate'}",
        ],
        metrics=metrics,
        x_parameter="sampling_ratio",
        output_dir=str(out_dir),
    )
    names = {os.path.basename(path) for path in result["plot_paths"]}
    for metric in metrics:
        png = out_dir / f"compare_{metric}_vs_sampling_ratio.png"
        assert png.is_file() and png.stat().st_size > 0
        assert png.name in names


def test_compare_defaults_skip_missing_dynamical_columns(tmp_path: Path):
    """Older aggregates that only have KLD / spectrum still overlay those."""
    rows = [
        {"sampling_ratio": "0.0", "kld_auto": "10.0", "spectrum_error_auto": "0.40"},
        {"sampling_ratio": "0.4", "kld_auto": "4.0", "spectrum_error_auto": "0.20"},
    ]
    csv_path = (
        tmp_path / "old" / "aggregate_eval_plots_sampling_ratio" / "aggregated_values.csv"
    )
    _write_csv(csv_path, rows)
    out_dir = tmp_path / "old_compare"
    result = run_compare(
        experiment_specs=[f"old|{tmp_path / 'old'}"],
        metrics=list(DEFAULT_COMPARE_METRICS),
        x_parameter="sampling_ratio",
        output_dir=str(out_dir),
    )
    names = {os.path.basename(path) for path in result["plot_paths"]}
    assert names == {
        "compare_kld_auto_vs_sampling_ratio.png",
        "compare_spectrum_error_auto_vs_sampling_ratio.png",
    }
    assert not (out_dir / "compare_lyap_max_vs_sampling_ratio.png").exists()
    assert (out_dir / "compare_kld_auto_vs_sampling_ratio.png").stat().st_size > 0


def test_channel_pair_missing_layout_returns_error(tmp_path: Path):
    csv_path = tmp_path / "plain" / "aggregated_values.csv"
    _write_csv(
        csv_path,
        [{"sampling_ratio": "0.1", "kld_auto": "1.0"}],
    )
    assert (
        main(
            [
                "--experiments",
                f"plain|{csv_path}",
                "--metrics",
                "kld_auto",
                "--channel-pair",
                "--output_dir",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )
