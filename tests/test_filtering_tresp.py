"""Tresp & Hofmann 1998 approximations on synthetic series."""

from __future__ import annotations

import numpy as np

from dvae.baselines.filtering.tresp import (
    TrespNonlinearModel,
    gibbs_impute,
    k_step_predictions,
    ml_substitute,
)


def _ar_predict(window: np.ndarray) -> np.ndarray:
    return np.array([0.8 * float(window.reshape(-1)[0])])


def test_ml_substitution_matches_the_linear_gaussian_mode():
    # Delay 1, one interior hole. Minimizing the two squared residuals that
    # touch y_t gives y = a (y_prev + y_next) / (1 + a^2).
    y = np.array([1.0, 0.5, 0.0, 0.2, 0.1])
    observed = np.array([True, True, False, True, True])
    filled = ml_substitute(_ar_predict, y, observed, delay=1)
    expected = 0.8 * (0.5 + 0.2) / (1.0 + 0.8**2)
    np.testing.assert_allclose(filled[2, 0], expected, atol=1e-4)
    np.testing.assert_allclose(filled[1, 0], 0.5)
    np.testing.assert_allclose(filled[3, 0], 0.2)


def test_gibbs_mean_is_near_the_conditional_mode():
    y = np.array([1.0, 0.5, 0.0, 0.2, 0.1])
    observed = np.array([[True], [True], [False], [True], [True]])
    rng = np.random.default_rng(0)
    samples = gibbs_impute(
        _ar_predict,
        y.reshape(-1, 1),
        observed,
        delay=1,
        sigma=0.05,
        n_sweeps=8,
        burn=2,
        n_sir=40,
        rng=rng,
    )
    expected = 0.8 * (0.5 + 0.2) / (1.0 + 0.8**2)
    np.testing.assert_allclose(samples[:, 2, 0].mean(), expected, atol=0.05)
    # Observed coordinates are clamped by the caller; the sampler may move
    # them internally only if they were marked missing. Index 1 stays put
    # because it is not in the missing set.
    np.testing.assert_allclose(samples[:, 1, 0], 0.5)


def test_k_step_sampling_matches_iteration_when_noise_is_zero():
    history = np.array([[0.4]])
    rng = np.random.default_rng(1)
    iterated, sampled = k_step_predictions(
        _ar_predict,
        history,
        n_steps=5,
        n_samples=30,
        sigma=0.0,
        rng=rng,
    )
    np.testing.assert_allclose(sampled, iterated, atol=1e-8)
    # y_k = 0.8^k * 0.4
    expected = 0.4 * (0.8 ** np.arange(1, 6))
    np.testing.assert_allclose(iterated[:, 0], expected, atol=1e-8)


def test_observed_target_training_recovers_ar_coefficient():
    rng = np.random.default_rng(0)
    t_len = 400
    y = np.zeros(t_len)
    for t in range(1, t_len):
        y[t] = 0.8 * y[t - 1] + rng.normal(0.0, 0.05)
    model = TrespNonlinearModel(
        obs_dim=1, delay=1, hidden=0, seed=0, weight_decay=0.0, standardize=True
    )
    model.fit(y, train_mode="observed_targets", epochs=80, lr=5e-2)
    weight = model.linear_weight()
    np.testing.assert_allclose(weight.reshape(-1)[0], 0.8, atol=0.08)
    assert model.train_mode == "observed_targets"
    assert model.sigma2 > 0


def test_ekf_on_masked_series_is_finite_and_clamps_observed_inputs():
    rng = np.random.default_rng(1)
    t_len = 80
    y = np.zeros(t_len)
    for t in range(1, t_len):
        y[t] = 0.7 * y[t - 1] + rng.normal(0.0, 0.05)
    missing = rng.random(t_len) < 0.3
    missing[:2] = False
    model = TrespNonlinearModel(obs_dim=1, delay=2, hidden=8, seed=1)
    model.fit(y, ~missing, train_mode="observed_targets", epochs=15, lr=1e-2)
    auto = np.zeros(t_len, dtype=bool)
    auto[t_len // 2 :] = True
    out = model.impute(
        y,
        ~missing,
        auto_mask=auto,
        do_ml=True,
        do_gibbs=True,
        gibbs_sweeps=2,
        gibbs_burn=1,
        n_sir=8,
        n_samples=4,
        n_particles=10,
    )
    for key in ("predicted", "filtered", "autonomous", "gibbs", "ml", "particle"):
        assert np.isfinite(out[key]).all(), key
    assert out["autonomous_sampled"].shape[0] == t_len // 2


def test_multiple_imputation_mode_runs():
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.normal(size=40)) * 0.1
    missing = np.zeros(len(y), dtype=bool)
    missing[5] = True
    missing[15] = True
    model = TrespNonlinearModel(obs_dim=1, delay=1, hidden=4, seed=2)
    model.fit(
        y,
        ~missing,
        train_mode="multiple_imputation",
        epochs=5,
        lr=1e-2,
        n_outer=2,
        gibbs_sweeps=1,
        gibbs_burn=0,
        n_sir=4,
    )
    assert model.train_mode == "multiple_imputation"
