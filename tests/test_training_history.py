"""Training-figure history: window stats, save, reload, and replot.

Synthetic arrays only. No dataset files and no model weights.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from dvae.dataset.gap_grid import (
    begin_window_recut,
    filter_window_starts,
    note_shift_stats,
    shift_starts_onto_observed,
    snapshot_window_stats,
)
from dvae.dataset.utils.data_utils import find_indices
from dvae.training_history import (
    DATASET_STAT_KEYS,
    NOT_PERSISTED_BEFORE,
    build_training_history,
    load_training_history,
    main,
    regenerate_training_figures,
    restore_epoch_series,
    save_training_history,
    series_or_new,
)
from dvae.visualizers.visualize_training_metrics import visualize_combined_metrics


def _series(n=4):
    epochs = np.arange(n, dtype=float)
    return {
        "train_loss": epochs + 1.0,
        "val_loss": epochs + 0.5,
        "train_recon": epochs + 0.4,
        "train_kl": epochs * 0.1,
        "val_recon": epochs + 0.2,
        "val_kl": epochs * 0.05,
        "val_spectrum": epochs + 0.3,
        "val_criteria_score": epochs + 2.0,
        "kl_warm_values": np.array([0.0, 0.0, 0.1, 0.2])[:n],
        "auto_warm_values": np.array([0.0, 0.1, 0.1, 0.2])[:n],
        "noise_warm_values": np.zeros(n),
        "sequence_len_values": np.array([80, 80, 100, 100])[:n],
        "cpt_patience_epochs": np.array([0, 1, 0, 1])[:n],
        "patience_threshold_epochs": np.array([2.0, 2.0, 6.0, 10.0])[:n],
        "delta_per_epoch": np.array([-0.2, 0.01, -0.3, 0.02])[:n],
        "best_state_epochs": np.array([0, 0, 2, 2])[:n],
        "windows_kept": np.array([12, 12, 8, 8])[:n],
        "windows_dropped_nan": np.array([3, 3, 1, 1])[:n],
        "windows_dropped_shift": np.array([2, 2, 0, 0])[:n],
        "gap_splits": np.array([4, 4, 2, 2])[:n],
        "mean_start_shift": np.array([1.5, 1.5, 0.0, 0.0])[:n],
    }


def _stats():
    return {
        "dataset_missing_rate": 0.125,
        "nan_ratio_threshold": 0.325,
        "nan_ratio_threshold_state": "value",
        "gap_quantile_len": 9.0,
        "gap_split_threshold": 20.0,
        "gap_split_threshold_state": "value",
        "block_mask_fraction": 0.05,
        "windows_kept": 8.0,
    }


def _history(n=4):
    series = _series(n)
    sigmas = np.arange(n * 2, dtype=float).reshape(2, n)
    return build_training_history(
        end=n,
        model_name="MT_VRNN",
        tag="unit",
        auto_sampling_method="ss",
        noise_sampling_method="none",
        delta_threshold=1e-4,
        val_criteria=("recon", "spectrum"),
        val_reference={"recon": 0.5, "spectrum": 0.3},
        series=series,
        dataset_stats=_stats(),
        sigmas_history=sigmas,
        a_sigmas_history=sigmas + 1.0,
    )


def test_new_history_stores_the_series_that_were_missing():
    history = _history()
    for key in NOT_PERSISTED_BEFORE:
        if key in DATASET_STAT_KEYS:
            assert key in history["dataset_stats"]
        else:
            assert key in history
            assert len(history[key]) == 4
    assert history["dataset_stats"]["dataset_missing_rate"] == 0.125
    assert history["dataset_stats"]["block_mask_fraction"] == 0.05
    assert history["val_criteria"] == "recon,spectrum"
    np.testing.assert_allclose(history["auto_warm_values"], [0.0, 0.1, 0.1, 0.2])
    np.testing.assert_allclose(history["patience_threshold_epochs"], [2, 2, 6, 10])
    assert history["sigmas_history"].shape == (2, 4)
    assert history["A_sigmas_history"].shape == (2, 4)
    # Change markers come from the saved prefix, with a leading 0.
    assert list(history["sequence_len_epochs"].tolist()) == [0, 2]


def test_save_reload_and_replot(tmp_path, monkeypatch):
    history = _history()
    path = tmp_path / "loss_model.pckl"
    save_training_history(path, history)
    loaded = load_training_history(path)
    np.testing.assert_allclose(loaded["windows_kept"], history["windows_kept"])
    np.testing.assert_allclose(
        loaded["mean_start_shift"], history["mean_start_shift"]
    )
    np.testing.assert_allclose(
        loaded["cpt_patience_epochs"], history["cpt_patience_epochs"]
    )
    assert loaded["dataset_stats"]["gap_quantile_len"] == 9.0
    assert loaded["dataset_stats"]["nan_ratio_threshold"] == 0.325
    # Raw spectrum, not the first-epoch ratio 0.3 / 0.3.
    np.testing.assert_allclose(loaded["val_spectrum"], [0.3, 1.3, 2.3, 3.3])
    assert loaded["val_spectrum"][0] == 0.3
    assert loaded["val_metric_reference"] == {"recon": 0.5, "spectrum": 0.3}
    assert loaded["spectrum_monitored"] is True
    assert loaded["val_criteria_reference"]["recon"] == 0.5
    assert loaded["val_criteria_reference"]["spectrum"] == 0.3

    restored = series_or_new(loaded, "sequence_len_values", 6, fill="hold")
    np.testing.assert_allclose(restored[:4], [80, 80, 100, 100])
    assert restored[4] == 100
    assert restored[5] == 100
    losses = series_or_new(loaded, "train_loss", 6, fill="zero")
    assert losses[4] == 0.0
    assert losses[5] == 0.0

    captured = {}
    real_savefig = plt.savefig
    real_fig_save = matplotlib.figure.Figure.savefig

    def _vertical_colors(ax):
        colors = []
        for line in ax.lines:
            x = np.asarray(line.get_xdata(), dtype=float)
            if x.size >= 2 and np.allclose(x, x[0]):
                colors.append(line.get_color())
        return colors

    def _save(fig_path, *args, **kwargs):
        fig = plt.gcf()
        if "delta_kl_cpt" in str(fig_path):
            cpt = fig.axes[3]
            dashed = [
                line
                for line in cpt.get_lines()
                if line.get_linestyle() == "--"
                and len(np.atleast_1d(line.get_ydata())) > 2
            ]
            captured["dashed"] = dashed
            captured["cpt_labels"] = [
                text.get_text() for text in cpt.get_legend().get_texts()
            ]
            windows = fig.axes[5]
            captured["window_labels"] = [
                text.get_text() for text in windows.get_legend().get_texts()
            ]
            captured["window_text"] = [text.get_text() for text in windows.texts]
            shift = fig.axes[6]
            captured["shift_label"] = shift.get_ylabel()
            captured["n_axes"] = len(fig.axes)
        return real_savefig(fig_path, *args, **kwargs)

    def _fig_save(self, fig_path, *args, **kwargs):
        name = str(fig_path)
        stem = Path(name).name
        if "val_criteria" in stem or stem.startswith("vis_training_val_drift"):
            is_log = stem.endswith("_log.png")
            is_varnorm = "_varnorm" in stem
            if stem.startswith("vis_training_val_drift"):
                key = "panels"
            elif "val_criteria_legacy" in stem:
                key = "legacy"
            elif "val_criteria_recononly" in stem:
                key = "recon"
            elif "val_criteria_nodrift" in stem:
                key = "nodrift"
            elif "val_criteria_drift" in stem:
                key = "drift"
            else:
                key = "both"
            if is_varnorm:
                key = key + "_varnorm"
            if is_log:
                key = key + "_log"
            captured[key] = {
                "titles": [
                    ax.get_legend().get_title().get_text() for ax in self.axes
                ],
                "ylabels": [ax.get_ylabel() for ax in self.axes],
                "best": [
                    float(np.asarray(line.get_ydata(), dtype=float).reshape(-1)[0])
                    for ax in self.axes
                    for line in ax.lines
                    if line.get_label() == "best epoch"
                ],
                "warmup": [_vertical_colors(ax) for ax in self.axes],
                "scales": [ax.get_yscale() for ax in self.axes],
                "ylims": [ax.get_ylim() for ax in self.axes],
                "labels": [
                    line.get_label()
                    for ax in self.axes
                    for line in ax.lines
                ],
                "bands": [
                    coll.get_label()
                    for ax in self.axes
                    for coll in ax.collections
                ],
            }
        return real_fig_save(self, fig_path, *args, **kwargs)

    monkeypatch.setattr(plt, "savefig", _save)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _fig_save)
    out = tmp_path / "figs"
    written = regenerate_training_figures(loaded, out)
    names = {Path(path).name for path in written}
    for name in (
        "vis_training_loss_unit.png",
        "vis_training_loss_unit_log.png",
        "vis_training_val_criteria_unit.png",
        "vis_training_loss_recon_unit.png",
        "vis_training_loss_recon_unit_log.png",
        "vis_training_loss_KLD_unit.png",
        "vis_training_loss_KLD_unit_log.png",
        "vis_training_delta_kl_cpt_best_state_unit.png",
        "vis_training_history_of_sigma_unit.png",
        "vis_training_history_of_alpha_unit.png",
        "vis_training_history_of_sigma_unit_diagA.png",
        "vis_training_history_of_alpha_unit_diagA.png",
    ):
        assert name in names
        assert (out / name).stat().st_size > 0
    assert captured["n_axes"] == 7
    assert "patience threshold" in captured["cpt_labels"]
    np.testing.assert_allclose(captured["dashed"][0].get_ydata(), [2, 2, 6, 10])
    assert captured["window_labels"] == [
        "windows kept",
        "dropped NaN",
        "dropped shift",
        "gap splits",
    ]
    text = "\n".join(captured["window_text"])
    assert "dataset missing rate: 0.125" in text
    assert "NaN-ratio threshold: 0.325" in text
    assert "p99 gap length: 9" in text
    assert "gap split threshold: 20" in text
    assert "block mask fraction: 0.05" in text
    assert captured["shift_label"] == "samples"
    both = captured["both"]
    assert both["titles"] == [
        "MT_VRNN: recon (selected)",
        "MT_VRNN: spectrum error (selected)",
        "MT_VRNN: selection score",
    ]
    assert both["ylabels"] == ["loss", "spectrum error", "normalised sum"]
    # best_state_epochs ends at 2: raw recon 2.5, raw spectrum 2.3, score 4.
    assert both["best"] == [2.5, 2.3, 4.0]
    for colors in both["warmup"]:
        assert "c" in colors
        assert "r" in colors
        assert "m" in colors

    recon_only = build_training_history(
        end=4,
        model_name="MT_VRNN",
        tag="recononly",
        auto_sampling_method="ss",
        noise_sampling_method="none",
        delta_threshold=1e-4,
        val_criteria=("recon",),
        val_reference={"recon": 0.5},
        metric_reference={"recon": 0.5, "spectrum": 0.3},
        spectrum_monitored=True,
        series=_series(),
        dataset_stats=_stats(),
    )
    recon_path = tmp_path / "recon_only.pckl"
    save_training_history(recon_path, recon_only)
    recon_loaded = load_training_history(recon_path)
    np.testing.assert_allclose(recon_loaded["val_spectrum"], [0.3, 1.3, 2.3, 3.3])
    assert recon_loaded["val_metric_reference"] == {"recon": 0.5, "spectrum": 0.3}
    assert recon_loaded["spectrum_monitored"] is True
    assert "val_criteria_reference" not in recon_loaded
    regenerate_training_figures(recon_loaded, tmp_path / "recon_figs")
    assert captured["recon"]["titles"] == [
        "MT_VRNN: recon (selected)",
        "MT_VRNN: spectrum error",
    ]
    assert captured["recon"]["best"] == [2.5, 2.3]
    assert "MT_VRNN: selection score" not in captured["recon"]["titles"]

    legacy = build_training_history(
        end=4,
        model_name="RNN",
        tag="legacy",
        auto_sampling_method="none",
        noise_sampling_method="none",
        delta_threshold=1e-4,
        val_criteria=("recon",),
        val_reference=None,
        metric_reference={"recon": 0.5},
        spectrum_monitored=False,
        series=_series(),
        dataset_stats=_stats(),
    )
    assert legacy["spectrum_monitored"] is False
    assert legacy["val_metric_reference"] == {"recon": 0.5}
    regenerate_training_figures(legacy, tmp_path / "legacy_figs")
    assert captured["legacy"]["titles"] == ["RNN: recon (selected)"]
    assert captured["legacy"]["best"] == [2.5]

    drift_series = _series()
    drift_series.update(
        {
            "val_base_error": np.array([1.0, 1.5, 2.0, 2.5]),
            "val_auto_drift": np.array([0.4, 0.4, 0.2, 0.2]),
            "val_error_drift_cross": np.array([0.1, 0.0, -0.1, 0.05]),
            "val_error_drift_total": np.array([1.5, 1.9, 2.1, 2.75]),
            "val_base_error_varnorm": np.array([0.5, 0.75, 1.0, 1.25]),
            "val_auto_drift_varnorm": np.array([0.2, 0.2, 0.1, 0.1]),
            "val_error_drift_cross_varnorm": np.array([0.05, 0.0, -0.05, 0.025]),
            "val_error_drift_total_varnorm": np.array([0.75, 0.95, 1.05, 1.375]),
            "val_auto_drift_k1": np.array([0.1, 0.1, 0.05, 0.05]),
            "val_auto_drift_k5": np.array([0.2, 0.2, 0.1, 0.1]),
            "val_auto_drift_k": np.array([0.4, 0.4, 0.2, 0.2]),
            "val_recon_k": np.array([1.5, 1.9, 2.1, 2.75]),
            "val_ftle": np.array([0.02, 0.03, 0.05, 0.04]),
            "val_lle_mean": np.array([0.40, 0.41, 0.42, 0.43]),
            "val_lle_std": np.array([0.01, 0.01, 0.02, 0.02]),
        }
    )
    lle_per_window = np.array(
        [
            [0.39, 0.41],
            [0.40, 0.42],
            [0.40, 0.44],
            [0.42, 0.44],
        ]
    )
    ftle_per_window = np.array(
        [
            [0.01, 0.03],
            [0.02, 0.04],
            [0.04, 0.06],
            [0.03, 0.05],
        ]
    )
    lle_gt_per_window = np.array([0.68, 0.70])
    window_info = {
        "starts": np.array([100, 2200]),
        "tf_observed": np.array([1.0, 0.97]),
        "tf_max_gap": np.array([0, 4]),
        "qualified": np.array([True, False]),
        "relaxed": True,
        "min_observed": 0.95,
        "max_gap_frac": 0.01,
        "seed": 1234567890,
    }
    drift_history = build_training_history(
        end=4,
        model_name="MT_VRNN",
        tag="drift",
        auto_sampling_method="ss",
        noise_sampling_method="none",
        delta_threshold=1e-4,
        val_criteria=("recon", "auto_drift"),
        val_reference={"recon": 0.5, "auto_drift": 0.4},
        metric_reference={
            "recon": 0.5,
            "spectrum": 0.3,
            "base_error": 1.0,
            "auto_drift": 0.4,
            "error_drift_cross": 0.1,
            "error_drift_total": 1.5,
        },
        spectrum_monitored=True,
        drift_monitored=True,
        drift_horizon=60,
        drift_flip=1000,
        x_true_var=np.array([0.5, 2.0]),
        window_info=window_info,
        lle_gt=0.69,
        rosenstein=True,
        ftle_fit=(1, 60),
        lle_per_window=lle_per_window,
        ftle_per_window=ftle_per_window,
        lle_gt_per_window=lle_gt_per_window,
        series=drift_series,
        dataset_stats=_stats(),
    )
    drift_path = tmp_path / "drift.pckl"
    save_training_history(drift_path, drift_history)
    drift_loaded = load_training_history(drift_path)
    np.testing.assert_allclose(drift_loaded["val_base_error"], [1.0, 1.5, 2.0, 2.5])
    np.testing.assert_allclose(
        drift_loaded["val_base_error_varnorm"], [0.5, 0.75, 1.0, 1.25]
    )
    np.testing.assert_allclose(
        drift_loaded["val_auto_drift"], drift_loaded["val_auto_drift_k"]
    )
    np.testing.assert_allclose(drift_loaded["val_auto_drift_k1"], [0.1, 0.1, 0.05, 0.05])
    np.testing.assert_allclose(drift_loaded["val_auto_drift_k5"], [0.2, 0.2, 0.1, 0.1])
    np.testing.assert_allclose(drift_loaded["val_recon_k"], drift_loaded["val_error_drift_total"])
    np.testing.assert_allclose(drift_loaded["val_ftle"], [0.02, 0.03, 0.05, 0.04])
    np.testing.assert_allclose(drift_loaded["val_lle_mean"], [0.40, 0.41, 0.42, 0.43])
    np.testing.assert_allclose(drift_loaded["val_lle_std"], [0.01, 0.01, 0.02, 0.02])
    assert drift_loaded["val_lle_gt"] == 0.69
    np.testing.assert_allclose(drift_loaded["val_lle_per_window"], lle_per_window)
    np.testing.assert_allclose(drift_loaded["val_ftle_per_window"], ftle_per_window)
    np.testing.assert_allclose(drift_loaded["val_lle_gt_per_window"], lle_gt_per_window)
    assert drift_loaded["val_rosenstein"] is True
    assert drift_loaded["val_ftle_fit_start"] == 1
    assert drift_loaded["val_ftle_fit_end"] == 60
    np.testing.assert_array_equal(drift_loaded["val_window_starts"], [100, 2200])
    np.testing.assert_allclose(drift_loaded["val_window_tf_observed"], [1.0, 0.97])
    np.testing.assert_array_equal(drift_loaded["val_window_tf_max_gap"], [0, 4])
    assert list(drift_loaded["val_window_qualified"]) == [True, False]
    assert drift_loaded["val_window_relaxed"] is True
    assert drift_loaded["val_window_tf_min_observed"] == 0.95
    assert drift_loaded["val_window_tf_max_gap_frac"] == 0.01
    assert drift_loaded["val_window_seed"] == 1234567890
    assert drift_loaded["val_base_error"][0] == 1.0
    assert drift_loaded["val_metric_reference"]["base_error"] == 1.0
    assert drift_loaded["val_metric_reference"]["auto_drift"] == 0.4
    np.testing.assert_allclose(drift_loaded["val_x_true_var"], [0.5, 2.0])
    assert drift_loaded["val_drift_horizon"] == 60
    assert drift_loaded["val_drift_flip"] == 1000
    assert drift_loaded["drift_monitored"] is True
    assert drift_loaded["val_criteria_reference"]["auto_drift"] == 0.4
    regenerate_training_figures(drift_loaded, tmp_path / "drift_figs")
    assert captured["drift"]["titles"] == [
        "MT_VRNN: recon (selected)",
        "MT_VRNN: spectrum error",
        "MT_VRNN: drift terms",
        "MT_VRNN: Lyapunov",
        "MT_VRNN: selection score",
    ]
    assert "recon_k" in captured["drift"]["labels"]
    assert "auto_drift (selected)" in captured["drift"]["labels"]
    assert "finite-time exponent" in captured["drift"]["labels"]
    assert "GT LLE" in captured["drift"]["labels"]
    assert "Rosenstein ± std" in captured["drift"]["bands"]
    assert captured["drift"]["best"] == [2.5, 2.3, 2.1, 0.05, 4.0]
    assert captured["drift_log"]["scales"] == ["symlog"] * 5
    assert captured["drift_varnorm"]["titles"] == [
        "MT_VRNN: recon (selected)",
        "MT_VRNN: spectrum error",
        "MT_VRNN: drift terms / var",
        "MT_VRNN: Lyapunov",
        "MT_VRNN: selection score",
    ]
    assert "recon_k / var" in captured["drift_varnorm"]["labels"]
    assert captured["drift_varnorm"]["best"] == [2.5, 2.3, 1.05, 0.05, 4.0]
    assert captured["drift_varnorm_log"]["scales"] == ["symlog"] * 5
    assert captured["panels"]["titles"] == [
        "MT_VRNN: base_error",
        "MT_VRNN: auto_drift (selected)",
        "MT_VRNN: error_drift_cross",
        "MT_VRNN: error_drift_total",
    ]
    assert captured["panels"]["ylims"][0] == captured["panels"]["ylims"][-1]
    assert captured["panels_log"]["scales"] == ["symlog"] * 4
    assert captured["panels_varnorm"]["titles"] == [
        "MT_VRNN: base_error / var",
        "MT_VRNN: auto_drift / var (selected)",
        "MT_VRNN: error_drift_cross / var",
        "MT_VRNN: error_drift_total / var",
    ]
    assert captured["panels_varnorm"]["best"] == [1.0, 0.1, -0.05, 1.05]
    assert captured["panels_varnorm_log"]["scales"] == ["symlog"] * 4
    drift_dir = tmp_path / "drift_figs"
    for name in (
        "vis_training_val_criteria_drift.png",
        "vis_training_val_criteria_drift_log.png",
        "vis_training_val_criteria_drift_varnorm.png",
        "vis_training_val_criteria_drift_varnorm_log.png",
        "vis_training_val_drift_drift.png",
        "vis_training_val_drift_drift_log.png",
        "vis_training_val_drift_drift_varnorm.png",
        "vis_training_val_drift_drift_varnorm_log.png",
    ):
        assert (drift_dir / name).is_file()
    for colors in captured["drift"]["warmup"]:
        assert "c" in colors and "r" in colors and "m" in colors

    hidden = build_training_history(
        end=4,
        model_name="RNN",
        tag="nodrift",
        auto_sampling_method="none",
        noise_sampling_method="none",
        delta_threshold=1e-4,
        val_criteria=("recon",),
        val_reference=None,
        metric_reference={"recon": 0.5},
        spectrum_monitored=False,
        drift_monitored=False,
        series=drift_series,
        dataset_stats=_stats(),
    )
    regenerate_training_figures(hidden, tmp_path / "nodrift_figs")
    assert captured["nodrift"]["titles"] == ["RNN: recon (selected)"]

    # The same redraw from a checkpoint loss_log, without using the weights.
    checkpoint = tmp_path / "MT_VRNN_checkpoint.pt"
    torch.save({"epoch": 1, "loss_log": loaded, "model_state_dict": {}}, checkpoint)
    from_pt = load_training_history(checkpoint)
    np.testing.assert_allclose(from_pt["auto_warm_values"], loaded["auto_warm_values"])
    cli_out = tmp_path / "cli"
    assert main([str(checkpoint), "--out-dir", str(cli_out), "--tag", "cli"]) == 0
    assert (cli_out / "vis_training_delta_kl_cpt_best_state_cli.png").is_file()
    assert (cli_out / "vis_training_loss_cli.png").is_file()


def test_reload_holds_warmup_values_and_zeros_losses():
    saved = np.array([0.2, 0.4])
    held = restore_epoch_series(saved, 4, fill="hold")
    np.testing.assert_allclose(held, [0.2, 0.4, 0.4, 0.4])
    padded = restore_epoch_series(saved, 4, fill="zero")
    np.testing.assert_allclose(padded, [0.2, 0.4, 0.0, 0.0])
    assert series_or_new({}, "auto_warm_values", 3, fill="hold").shape == (3,)


def test_old_loss_pickle_still_replots(tmp_path):
    old = {
        "train_loss": np.ones(3),
        "val_loss": np.full(3, 2.0),
        "kl_warm_epochs": np.array([0]),
    }
    for key in NOT_PERSISTED_BEFORE:
        assert key not in old
    out = tmp_path / "old"
    written = regenerate_training_figures(old, out, tag="old")
    names = {Path(path).name for path in written}
    assert "vis_training_loss_old.png" in names
    assert "vis_training_delta_kl_cpt_best_state_old.png" in names
    assert "vis_training_loss_recon_old.png" not in names


def test_shift_and_nan_recut_stats_are_recorded():
    observed = np.ones(20, dtype=bool)
    observed[0] = False
    observed[3:6] = False
    stats = {}
    got = shift_starts_onto_observed(
        [0, 3, 8], observed, 4, None, 20, shift_stats=stats
    )
    assert list(got.tolist()) == [1, 6, 8]
    assert stats["dropped_unfit"] == 0
    assert stats["shift_count"] == 3
    assert stats["shift_sum"] == 4.0

    blocked = np.ones(20, dtype=bool)
    blocked[6:10] = False
    dropped_stats = {}
    dropped = shift_starts_onto_observed(
        [6], blocked, 4, [10], 20, shift_stats=dropped_stats
    )
    assert list(dropped.tolist()) == []
    assert dropped_stats["dropped_unfit"] == 1
    assert dropped_stats["shift_count"] == 0

    flags = np.ones(30, dtype=bool)
    flags[:5] = False
    strict_stats = {}
    starts = find_indices(
        30,
        10,
        1,
        strict_non_overlap=True,
        observed=flags,
        shift_stats=strict_stats,
    )
    assert int(starts[0]) == 5
    assert strict_stats["dropped_unfit"] == 0
    assert strict_stats["shift_sum"] == 5.0

    class _Rows:
        seq = np.array([[0.0], [np.nan], [0.0], [0.0]])
        seq_len = 2
        dataset_missing_rate = 0.25
        max_window_nan_ratio = 0.4
        max_window_nan_ratio_off = False
        legacy_windowing = False

    rows = _Rows()
    begin_window_recut(rows)
    note_shift_stats(rows, dropped_stats)
    kept = filter_window_starts(rows, [0, 1, 2])
    assert list(kept.tolist()) == [2]
    snap = snapshot_window_stats(rows)
    assert snap["windows_kept"] == 1
    assert snap["windows_dropped_nan"] == 2
    assert snap["windows_dropped_shift"] == 1
    assert snap["nan_ratio_threshold"] == 0.4
    assert snap["nan_ratio_threshold_state"] == "value"
    assert snap["dataset_missing_rate"] == 0.25


def test_combined_figure_draws_window_steps_without_a_caller_series(tmp_path):
    """Panels stay on when a caller has not supplied window arrays yet."""
    n = 3
    visualize_combined_metrics(
        np.zeros(n),
        1e-4,
        np.array([0]),
        np.zeros(n),
        np.array([0]),
        np.zeros(n),
        np.array([0]),
        np.zeros(n),
        np.array([0]),
        np.full(n, 40),
        np.arange(n),
        np.zeros(n),
        "PLRNN",
        "ss",
        "none",
        str(tmp_path),
        "bare",
        np.full(n, 5.0),
    )
    assert (tmp_path / "vis_training_delta_kl_cpt_best_state_bare.png").stat().st_size > 0
