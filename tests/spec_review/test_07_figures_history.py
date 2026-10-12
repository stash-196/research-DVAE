"""Spec 7. The validation figure and the on-disk history.

Raw recon and raw spectrum error are plotted even when neither is the
selection criterion. Windowing stats, the patience threshold, timings,
and bump_events are in the history. The replot CLI redraws from that
file alone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from dvae.epoch_log import make_bump_row
from dvae.training_history import (
    build_training_history,
    load_training_history,
    main,
    save_training_history,
)
from dvae.visualizers.visualize_training_metrics import (
    val_criteria_plot_series,
    visualize_combined_metrics,
    visualize_val_criteria,
)


def _series(n=3):
    epochs = np.arange(n, dtype=float)
    return {
        "train_loss": epochs + 1.0,
        "val_loss": epochs + 2.0,
        "train_recon": epochs + 0.4,
        "train_kl": epochs * 0.1,
        "val_recon": epochs + 0.2,
        "val_kl": epochs * 0.05,
        "val_spectrum": epochs + 3.0,
        "val_criteria_score": epochs + 4.0,
        "kl_warm_values": np.zeros(n),
        "auto_warm_values": np.zeros(n),
        "noise_warm_values": np.zeros(n),
        "sequence_len_values": np.full(n, 20.0),
        "cpt_patience_epochs": np.arange(n, dtype=float),
        "patience_threshold_epochs": np.full(n, 40.0),
        "delta_per_epoch": np.zeros(n),
        "best_state_epochs": np.zeros(n),
        "windows_kept": np.full(n, 8.0),
        "windows_dropped_nan": np.full(n, 1.0),
        "windows_dropped_shift": np.full(n, 2.0),
        "gap_splits": np.full(n, 3.0),
        "mean_start_shift": np.full(n, 1.5),
        "time_train": np.full(n, 0.4),
        "time_val": np.full(n, 0.2),
        "time_val_forward": np.full(n, 0.05),
        "time_spectrum": np.full(n, 0.01),
        "time_drift": np.full(n, 0.02),
        "time_rosenstein": np.full(n, 0.03),
        "time_plot": np.full(n, 0.07),
        "time_eval": np.array([0.0, 0.0, 0.2]),
        "val_ftle": epochs + 0.01,
        "val_base_error": epochs + 0.1,
        "val_auto_drift": epochs + 0.2,
        "val_error_drift_cross": epochs - 0.3,
        "val_error_drift_total": epochs + 0.4,
        "val_recon_k": epochs + 0.4,
        "val_lle_mean": np.full(n, 0.9),
        "val_lle_std": np.full(n, 0.01),
    }


def _history():
    info = {
        "starts": np.array([10, 40]),
        "tf_observed": np.array([1.0, 0.99]),
        "tf_max_gap": np.array([0, 1]),
        "qualified": np.array([True, True]),
        "relaxed": False,
        "min_observed": 0.95,
        "max_gap_frac": 0.01,
        "seed": 1234567890,
        "tf_steps": 1000,
    }
    return build_training_history(
        end=3,
        model_name="VRNN",
        tag="synthetic",
        auto_sampling_method="ss",
        noise_sampling_method="none",
        delta_threshold=1e-4,
        val_criteria=("ftle",),
        val_reference={"ftle": 0.5},
        series=_series(),
        dataset_stats={
            "dataset_missing_rate": 0.05,
            "nan_ratio_threshold": 0.25,
            "gap_quantile_len": 4.0,
            "gap_split_threshold": 8.0,
            "block_mask_fraction": 0.0,
        },
        spectrum_monitored=True,
        drift_monitored=True,
        drift_horizon=60,
        drift_flip=1000,
        window_info=info,
        bump_events=[
            make_bump_row(1, "window", 10, 12, 40, 40, 41),
        ],
    )


def test_validation_figure_plots_raw_recon_and_spectrum_for_any_criterion():
    series = val_criteria_plot_series(
        ("ftle",),
        val_loss=np.array([2.0, 3.0]),
        val_spectrum=np.array([4.0, 5.0]),
        val_criteria_score=None,
        include_spectrum=True,
    )
    names = [item[0] for item in series]
    assert names[:2] == ["recon", "spectrum"]
    np.testing.assert_array_equal(series[0][1], [2.0, 3.0])
    np.testing.assert_array_equal(series[1][1], [4.0, 5.0])
    assert series[0][2] is False
    assert series[1][2] is False


def test_history_persists_plotted_series_and_replot_uses_only_the_file(tmp_path, monkeypatch):
    history = _history()
    path = tmp_path / "loss_model.pckl"
    save_training_history(path, history)
    loaded = load_training_history(path)
    for key in (
        "val_loss",
        "val_spectrum",
        "windows_kept",
        "windows_dropped_nan",
        "windows_dropped_shift",
        "gap_splits",
        "mean_start_shift",
        "patience_threshold_epochs",
        "time_train",
        "time_val",
        "time_val_forward",
        "time_spectrum",
        "time_drift",
        "time_rosenstein",
        "time_plot",
        "time_eval",
    ):
        assert key in loaded
        assert len(np.asarray(loaded[key]).reshape(-1)) == 3
    assert loaded["bump_events"][0]["type"] == "window"
    assert loaded["bump_events"][0]["epoch"] == 1
    assert loaded["val_criteria"] == "ftle"
    np.testing.assert_array_equal(loaded["val_window_starts"], [10, 40])
    assert loaded["val_drift_horizon"] == 60
    assert loaded["val_drift_flip"] == 1000
    assert loaded["dataset_stats"]["nan_ratio_threshold"] == 0.25

    recorded = {}
    real_val = visualize_val_criteria
    real_combined = visualize_combined_metrics

    def _val(*args, **kwargs):
        recorded["criteria"] = args[0]
        recorded["val_loss"] = np.asarray(args[1], dtype=float).copy()
        recorded["val_spectrum"] = np.asarray(args[2], dtype=float).copy()
        recorded["include_spectrum"] = kwargs.get("include_spectrum")
        return real_val(*args, **kwargs)

    def _combined(*args, **kwargs):
        # patience_threshold_epochs is passed positionally by the replot path.
        recorded["patience"] = args[17]
        recorded["windows_kept"] = kwargs.get("windows_kept")
        recorded["timing"] = kwargs.get("timing")
        recorded["gap_splits"] = kwargs.get("gap_splits")
        return real_combined(*args, **kwargs)

    # regenerate imports these names when it runs, so patch the source module.
    monkeypatch.setattr(
        "dvae.visualizers.visualize_training_metrics.visualize_val_criteria",
        _val,
    )
    monkeypatch.setattr(
        "dvae.visualizers.visualize_training_metrics.visualize_combined_metrics",
        _combined,
    )
    out = tmp_path / "figures"
    status = main([str(path), "--out-dir", str(out)])
    assert status == 0
    assert list(out.glob("*.png"))
    assert tuple(recorded["criteria"]) == ("ftle",)
    np.testing.assert_array_equal(recorded["val_loss"], loaded["val_loss"])
    np.testing.assert_array_equal(recorded["val_spectrum"], loaded["val_spectrum"])
    assert recorded["include_spectrum"] is True
    np.testing.assert_array_equal(
        recorded["patience"], loaded["patience_threshold_epochs"]
    )
    np.testing.assert_array_equal(recorded["windows_kept"], loaded["windows_kept"])
    np.testing.assert_array_equal(recorded["gap_splits"], loaded["gap_splits"])
    assert recorded["timing"]["time_train"][0] == loaded["time_train"][0]
