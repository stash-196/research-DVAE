"""Spec 6. Lyapunov scores and validation-criterion selection.

FTLE is the slope of log ||d_k||. Rosenstein is scored on the free-run
part. ftle and rosenstein_lle select |estimate - GT|. The cross term
selects |cross|. Every new metric is a val_criteria name. Two or more
criteria are divided by the first validation epoch and summed.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from dvae.eval.utils.local_drift_analysis import (
    drift_metrics_from_series,
    finite_time_exponent,
    rosenstein_on_free_run,
)
from dvae.learning_algo import LearningAlgorithm
from dvae.run_options import (
    SELECTABLE_VAL_CRITERIA,
    normalised_criterion_sum,
    parse_val_criteria,
    selection_criterion_value,
)


def test_ftle_is_the_slope_of_log_norm():
    slope = 0.05
    horizon = 60
    norms = np.exp(slope * np.arange(1, horizon + 1))
    assert finite_time_exponent(norms, k_start=1, k_end=horizon) == pytest.approx(
        slope
    )

    length = 80
    flip = 20
    truth = np.zeros((length, 1))
    teacher = np.zeros_like(truth)
    autonomous = teacher.copy()
    for k in range(1, horizon + 1):
        autonomous[flip + k - 1, 0] = np.exp(slope * k)
    metrics = drift_metrics_from_series(
        [truth],
        [teacher],
        [autonomous],
        flip=flip,
        horizon=horizon,
        variance=np.array([1.0]),
        ftle_k_start=1,
        ftle_k_end=horizon,
    )
    assert metrics["ftle"] == pytest.approx(slope)
    assert metrics["ftle_per_window"][0] == pytest.approx(slope)


def test_rosenstein_sees_only_the_free_run_part(monkeypatch):
    seen = []

    def _fake(series, m, tau, dt):
        seen.append(np.asarray(series, dtype=float).copy())
        return 0.25

    monkeypatch.setattr(
        "dvae.eval.utils.rosenstein_lyapunov.rosenstein_lle",
        _fake,
    )
    window = np.arange(30, dtype=float).reshape(-1, 1)
    mean, std, per_window = rosenstein_on_free_run([window], tf_steps=10)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], window[10:, 0])
    assert mean == pytest.approx(0.25)
    assert per_window[0] == pytest.approx(0.25)
    assert std == pytest.approx(0.0)


def test_new_metrics_are_selectable_as_absolute_cross_and_lyapunov_distance():
    required = {
        "recon",
        "spectrum",
        "base_error",
        "auto_drift",
        "error_drift_cross",
        "error_drift_total",
        "recon_k",
        "base_error_varnorm",
        "auto_drift_varnorm",
        "error_drift_cross_varnorm",
        "error_drift_total_varnorm",
        "auto_drift_k1",
        "auto_drift_k5",
        "auto_drift_k",
        "ftle",
        "rosenstein_lle",
    }
    assert required <= set(SELECTABLE_VAL_CRITERIA)
    parsed = parse_val_criteria(
        "drift,cross,auto_total,ftle,rosenstein_lle,recon_k"
    )
    assert parsed == (
        "auto_drift",
        "error_drift_cross",
        "error_drift_total",
        "ftle",
        "rosenstein_lle",
        "recon_k",
    )
    assert selection_criterion_value("error_drift_cross", -1.25) == pytest.approx(
        1.25
    )
    assert selection_criterion_value(
        "error_drift_cross_varnorm", 0.4
    ) == pytest.approx(0.4)
    assert selection_criterion_value("ftle", 1.7, gt_lle=1.0) == pytest.approx(0.7)
    assert selection_criterion_value(
        "rosenstein_lle", 0.2, gt_lle=0.5
    ) == pytest.approx(0.3)


def test_multiple_criteria_are_normalised_by_the_first_validation_epoch():
    algo = LearningAlgorithm.__new__(LearningAlgorithm)
    algo.logger = logging.getLogger("spec-review-selection")
    algo._spectrum_windows = None
    algo._val_criteria = ("ftle", "error_drift_cross", "rosenstein_lle")
    algo._val_reference = None
    algo._metric_reference = None
    algo._lle_gt = 1.0
    val_loss = np.array([5.0, 4.0])
    val_spectrum = np.array([np.nan, np.nan])
    score = np.array([np.nan, np.nan])
    drift = {
        "val_ftle": np.array([1.5, 1.2]),
        "val_error_drift_cross": np.array([-0.4, 0.8]),
        "val_lle_mean": np.array([1.4, 0.5]),
    }
    first, raw, norm = algo._epoch_selection_value(
        0, val_loss, val_spectrum, score, drift
    )
    assert raw["ftle"] == pytest.approx(0.5)
    assert raw["error_drift_cross"] == pytest.approx(0.4)
    assert raw["rosenstein_lle"] == pytest.approx(0.4)
    assert norm["ftle"] == pytest.approx(1.0)
    assert norm["error_drift_cross"] == pytest.approx(1.0)
    assert norm["rosenstein_lle"] == pytest.approx(1.0)
    assert first == pytest.approx(3.0)

    second, raw2, norm2 = algo._epoch_selection_value(
        1, val_loss, val_spectrum, score, drift
    )
    assert raw2["ftle"] == pytest.approx(0.2)
    assert raw2["error_drift_cross"] == pytest.approx(0.8)
    assert raw2["rosenstein_lle"] == pytest.approx(0.5)
    assert norm2["ftle"] == pytest.approx(0.2 / 0.5)
    assert norm2["error_drift_cross"] == pytest.approx(0.8 / 0.4)
    assert norm2["rosenstein_lle"] == pytest.approx(0.5 / 0.4)
    assert second == pytest.approx(normalised_criterion_sum(raw2, algo._val_reference))

    algo._val_criteria = ("recon",)
    algo._val_reference = None
    alone, raw_alone, norm_alone = algo._epoch_selection_value(
        1, val_loss, val_spectrum, score, drift
    )
    assert alone == pytest.approx(4.0)
    assert norm_alone == {}
