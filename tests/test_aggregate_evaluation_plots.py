"""Plots for Lyapunov, Jacobian, local-drift, and intrinsic-dim scalars."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import yaml

from dvae.eval.aggregate_evaluation_results import (
    build_aggregated_values_table,
    discover_scalar_metrics,
    finite_float,
    flatten_scalar_fields,
    is_auto_plot_metric_key,
    plot_1d,
    plot_2d,
    plot_parameter_metrics,
    resolve_metrics_to_plot,
)
from dvae.eval.aggregate_plot_style import (
    DEFAULT_AGGREGATE_METRICS,
    get_metric_display_name,
    resolve_heatmap_limits,
)


def _run(sampling_ratio, mask_label, **metrics):
    return {
        "yaml_file": (
            f"/tmp/run_s{sampling_ratio}_m{mask_label}/evaluation_summary.yaml"
        ),
        "params": {
            "sampling_ratio": sampling_ratio,
            "mask_label": mask_label,
        },
        **metrics,
    }


def _score_block(scale):
    """Top-level scalars as eval_signal / jacobian_lyapunov write them."""
    return {
        "kld_auto": 4.0 * scale,
        "spectrum_error_auto": 0.2 * scale,
        "lyap_max": 0.35 * scale,
        "lyap_spectrum": [0.35 * scale, -0.12, -0.40],
        "lyap_n_steps": 1000,
        "lyap_n_transient": 100,
        "lyap_n_warmup": 50,
        "lyap_k": 3,
        "lyap_map": "MT_RNN:shPLRNN:analytic_shPLRNN",
        "lyap_autonomous_input": 0,
        "jac_opnorm_mean": 1.1 * scale,
        "jac_opnorm_max": 1.8 * scale,
        "jac_rho_max": 1.05 * scale,
        "jac_rho_gt1_frac": 0.25 * scale,
        "local_drift_avg_d_norm": 0.04 * scale,
        "local_drift_avg_d_norm_std": 0.01 * scale,
        "local_drift_avg_cross_term": -0.02 * scale,
        "local_drift_avg_cross_term_std": 0.005,
        "local_drift_avg_delta_mse": 0.008 * scale,
        "local_drift_avg_delta_mse_std": 0.002,
        "local_drift_per_step_d_norm_mean": [0.01, 0.02, 0.03],
        "local_drift_flip_point": 30,
        "local_drift_auto_len": 30,
        "local_drift_n_batches": 4,
        "id_pr_gt": 2.1,
        "id_pr_tf": 2.0,
        "id_pr_auto": 2.4 * scale,
        "id_twonn_gt": 1.8,
        "id_twonn_tf": 1.7,
        "id_twonn_auto": 1.9,
        "id_pr_joint_gt": 3.2,
        "id_pr_joint_auto": 3.5,
        "id_twonn_joint_gt": 2.4,
        "id_pr_gt_ch1": 2.0,
        "id_pr_gt_ch2": 2.6,
        "id_twonn_auto_ch1": 1.7,
        "id_pr_gt_std_across_batches": 0.15,
        "id_pr_hidden_joint_tf": 3.2 * scale,
        "id_pr_hidden_joint_auto": 4.1 * scale,
        "id_twonn_hidden_joint_tf": 2.8 * scale,
        "id_twonn_hidden_joint_auto": 3.6 * scale,
    }


def _table():
    return [
        _run(0.0, 0.1, **_score_block(1.0)),
        _run(0.4, 0.1, **_score_block(0.8)),
        _run(0.0, 0.5, **_score_block(1.2)),
        _run(0.4, 0.5, **_score_block(0.6)),
    ]


def test_flatten_dumps_lyapunov_spectrum_and_prefixes_nested_blocks():
    flat = {}
    flatten_scalar_fields(
        "",
        {
            "lyap_max": 0.25,
            "lyap_spectrum": [0.25, -0.1, -0.4],
            "jac_opnorm_mean": 1.5,
            "local_drift_per_step_d_norm_mean": [0.01, 0.02],
        },
        flat,
    )
    assert flat["lyap_max"] == 0.25
    assert flat["jac_opnorm_mean"] == 1.5
    assert isinstance(flat["lyap_spectrum"], str)
    assert flat["lyap_spectrum"].startswith("[")
    assert "lyap_spectrum_0" not in flat
    assert isinstance(flat["local_drift_per_step_d_norm_mean"], str)
    assert finite_float(flat["lyap_spectrum"]) is None

    nested = {}
    flatten_scalar_fields(
        "lyapunov",
        {"lyap_max": 0.3, "lyap_spectrum": [0.3, -0.2]},
        nested,
    )
    assert nested["lyapunov_lyap_max"] == 0.3
    assert "lyap_max" not in nested
    parsed = yaml.safe_load(nested["lyapunov_lyap_spectrum"])
    assert parsed[0] == 0.3


def test_discover_keeps_scalar_families_and_skips_lists_and_settings():
    data = _table()
    found = discover_scalar_metrics(data)
    for key in (
        "lyap_max",
        "jac_opnorm_mean",
        "jac_opnorm_max",
        "jac_rho_max",
        "jac_rho_gt1_frac",
        "local_drift_avg_d_norm",
        "local_drift_avg_d_norm_std",
        "local_drift_avg_cross_term",
        "local_drift_avg_delta_mse",
        "id_pr_gt",
        "id_pr_tf",
        "id_pr_auto",
        "id_twonn_gt",
        "id_twonn_auto",
        "id_pr_joint_gt",
        "id_pr_joint_auto",
        "id_twonn_joint_gt",
        "id_pr_gt_ch1",
        "id_pr_gt_ch2",
        "id_twonn_auto_ch1",
    ):
        assert key in found
    for key in (
        "lyap_spectrum",
        "lyap_n_steps",
        "lyap_map",
        "lyap_autonomous_input",
        "local_drift_per_step_d_norm_mean",
        "local_drift_flip_point",
        "local_drift_n_batches",
        "id_pr_gt_std_across_batches",
        "kld_auto",
    ):
        assert key not in found
    assert is_auto_plot_metric_key("lyap_spectrum") is False
    assert is_auto_plot_metric_key("id_pr_gt_ch1") is True

    nested_only = [
        {
            "yaml_file": "/tmp/nested/evaluation_summary.yaml",
            "params": {"sampling_ratio": 0.2},
            "lyapunov": {"lyap_max": 0.3, "lyap_spectrum": [0.3, -0.1]},
        }
    ]
    assert "lyap_max" not in discover_scalar_metrics(nested_only)
    assert discover_scalar_metrics(nested_only) == []


def test_old_summaries_skip_missing_dynamical_columns():
    old = [
        _run(0.0, 0.0, kld_auto=3.0, spectrum_error_auto=0.2),
        _run(0.5, 0.0, kld_auto=1.0, spectrum_error_auto=0.1),
    ]
    metrics = resolve_metrics_to_plot(list(DEFAULT_AGGREGATE_METRICS), old)
    assert metrics == ["kld_auto", "spectrum_error_auto"]
    assert "lyap_max" not in metrics
    assert "id_pr_gt" not in metrics


def test_resolve_adds_present_id_and_drift_std_keys():
    metrics = resolve_metrics_to_plot(list(DEFAULT_AGGREGATE_METRICS), _table())
    assert metrics[0] == "kld_auto" or "kld_auto" in metrics
    for key in DEFAULT_AGGREGATE_METRICS:
        if key in ("kld_tf", "spectrum_error_gt", "spectrum_error_tf"):
            assert key not in metrics
        else:
            assert key in metrics
    assert "id_pr_gt" in metrics
    assert "id_pr_gt_ch1" in metrics
    assert "id_twonn_joint_gt" in metrics
    assert "local_drift_avg_d_norm_std" in metrics
    assert "lyap_spectrum" not in metrics
    assert "id_pr_gt_std_across_batches" not in metrics
    rows = build_aggregated_values_table(_table(), ["sampling_ratio"])
    assert rows[0]["lyap_max"] == 0.35
    assert isinstance(rows[0]["lyap_spectrum"], str)


def test_display_names_and_heatmap_limits():
    assert get_metric_display_name("lyap_max") == "Max Lyapunov exponent"
    assert get_metric_display_name("jac_opnorm_mean") == (
        "Jacobian operator norm (mean)"
    )
    assert get_metric_display_name("jac_opnorm_max") == (
        "Jacobian operator norm (max)"
    )
    assert get_metric_display_name("jac_rho_gt1_frac") == (
        "Fraction of steps with rho(J) > 1"
    )
    assert get_metric_display_name("local_drift_avg_d_norm") == (
        "Local drift mean ||d||^2"
    )
    assert get_metric_display_name("local_drift_avg_cross_term") == (
        "Local drift mean d^T e"
    )
    assert get_metric_display_name("local_drift_avg_delta_mse") == (
        "Local drift mean delta MSE"
    )
    assert get_metric_display_name("local_drift_avg_d_norm_std") == (
        "Local drift mean ||d||^2 (std)"
    )
    assert get_metric_display_name("id_pr_gt") == "PR intrinsic dim (GT)"
    assert get_metric_display_name("id_pr_gt_ch1") == "PR intrinsic dim (GT) (ch1)"
    assert get_metric_display_name("id_twonn_auto") == "TwoNN intrinsic dim (Auto)"
    assert get_metric_display_name("id_pr_joint_gt") == (
        "PR intrinsic dim joint (GT)"
    )
    assert get_metric_display_name("id_twonn_joint_auto") == (
        "TwoNN intrinsic dim joint (Auto)"
    )
    assert get_metric_display_name("id_pr_hidden_joint_tf") == (
        "PR intrinsic dim hidden joint (TF)"
    )
    assert get_metric_display_name("id_twonn_hidden_joint_auto") == (
        "TwoNN intrinsic dim hidden joint (Auto)"
    )
    assert get_metric_display_name("id_pr_gt_std_across_batches") == (
        "PR intrinsic dim (GT) (std across batches)"
    )

    assert resolve_heatmap_limits("spectrum_error_auto") == (0.0, 1.0, "linear")
    assert resolve_heatmap_limits("kld_auto") == (-10.0, 1e3, "symlog")
    assert resolve_heatmap_limits("jac_rho_gt1_frac") == (0.0, 1.0, "linear")
    assert resolve_heatmap_limits("id_pr_auto") == (0.0, 20.0, "linear")
    assert resolve_heatmap_limits("id_pr_gt_ch2") == (0.0, 20.0, "linear")
    assert resolve_heatmap_limits("id_twonn_joint_gt") == (0.0, 20.0, "linear")
    assert resolve_heatmap_limits("id_pr_hidden_joint_tf") is None
    assert resolve_heatmap_limits("id_twonn_hidden_joint_auto") is None
    assert resolve_heatmap_limits("id_pr_gt_std_across_batches") is None
    assert resolve_heatmap_limits("lyap_max") is None
    assert resolve_heatmap_limits("jac_opnorm_mean") is None
    assert resolve_heatmap_limits("local_drift_avg_cross_term") is None


def test_plot_path_writes_heatmaps_and_graphs(tmp_path: Path):
    data = _table()
    metrics = resolve_metrics_to_plot(list(DEFAULT_AGGREGATE_METRICS), data)
    out = tmp_path / "plots"
    out.mkdir()
    plot_parameter_metrics(
        data,
        ["sampling_ratio", "mask_label"],
        metrics,
        str(out),
    )
    expected = [
        "lyap_max",
        "jac_opnorm_mean",
        "jac_opnorm_max",
        "jac_rho_max",
        "jac_rho_gt1_frac",
        "local_drift_avg_d_norm",
        "local_drift_avg_cross_term",
        "local_drift_avg_delta_mse",
        "local_drift_avg_d_norm_std",
        "id_pr_gt",
        "id_pr_auto",
        "id_twonn_gt",
        "id_pr_joint_gt",
        "id_pr_gt_ch1",
        "id_pr_hidden_joint_tf",
        "id_pr_hidden_joint_auto",
        "id_twonn_hidden_joint_tf",
        "id_twonn_hidden_joint_auto",
        "kld_auto",
        "spectrum_error_auto",
    ]
    for metric in expected:
        for name in (
            f"{metric}_vs_sampling_ratio.png",
            f"{metric}_vs_mask_label.png",
            f"{metric}_heatmap_sampling_ratio_vs_mask_label.png",
        ):
            path = out / name
            assert path.is_file(), name
            assert path.stat().st_size > 0

    names = {path.name for path in out.iterdir()}
    assert not any("lyap_spectrum" in name for name in names)
    assert not any("per_step" in name for name in names)
    assert not any("std_across_batches" in name for name in names)
    assert not any(name.startswith("lyap_n_") for name in names)
    assert not any(name.startswith("lyap_map") for name in names)


def test_single_parameter_heatmap_and_spectrum_list_skip(tmp_path: Path):
    data = [
        _run(0.1, 0.0, lyap_max=0.2, lyap_spectrum=[0.2, -0.1], id_pr_gt=2.5),
        _run(0.8, 0.0, lyap_max=-0.1, lyap_spectrum=[-0.1, -0.4], id_pr_gt=1.5),
    ]
    out = tmp_path / "one"
    out.mkdir()
    plot_1d(data, "sampling_ratio", "lyap_spectrum", str(out))
    plot_2d(data, "sampling_ratio", "lyap_spectrum", str(out))
    assert list(out.iterdir()) == []

    plot_parameter_metrics(data, ["sampling_ratio"], ["lyap_max", "id_pr_gt"], str(out))
    for name in (
        "lyap_max_vs_sampling_ratio.png",
        "lyap_max_heatmap_sampling_ratio.png",
        "id_pr_gt_vs_sampling_ratio.png",
        "id_pr_gt_heatmap_sampling_ratio.png",
    ):
        path = out / name
        assert path.is_file() and path.stat().st_size > 0


def test_heatmap_warns_when_fraction_exceeds_unit_interval(tmp_path: Path, capsys):
    data = [
        _run(0.0, 0.0, jac_rho_gt1_frac=1.5),
        _run(0.5, 0.0, jac_rho_gt1_frac=0.2),
    ]
    plot_2d(data, "sampling_ratio", "jac_rho_gt1_frac", str(tmp_path))
    captured = capsys.readouterr().out
    assert "jac_rho_gt1_frac" in captured
    assert "clip outside fixed range" in captured
    assert (tmp_path / "jac_rho_gt1_frac_heatmap_sampling_ratio.png").is_file()
