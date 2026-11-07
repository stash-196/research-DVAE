"""Kalman filter / RTS smoother / EM under missing observations.

Synthetic linear-Gaussian series only. The smoother is checked against the
exact conditional mean of the joint Gaussian, and EM is checked by recovery
of an identifiable system (H fixed at I).
"""

from __future__ import annotations

import numpy as np
import pytest

from dvae.baselines.filtering.kalman_em import (
    LinearGaussianEM,
    simulate_with_initial_cov,
)


def _batch_posterior_mean(model, y, observed):
    """Exact E[x | y_obs] for the linear-Gaussian model, by conditioning."""
    f = model.F
    h = model.H
    q = model.Q
    r = model.R
    mu0 = model.mu0
    p0 = model.P0
    t_len, obs_dim = y.shape
    state_dim = f.shape[0]
    mu = np.zeros((t_len, state_dim), dtype=np.float64)
    mu[0] = mu0
    var = np.zeros((t_len, state_dim, state_dim), dtype=np.float64)
    var[0] = p0
    for t in range(1, t_len):
        mu[t] = f @ mu[t - 1]
        var[t] = f @ var[t - 1] @ f.T + q

    dim = t_len * state_dim
    sxx = np.zeros((dim, dim), dtype=np.float64)
    eye_cache = {0: np.eye(state_dim)}

    def f_power(k):
        if k not in eye_cache:
            eye_cache[k] = np.linalg.matrix_power(f, k)
        return eye_cache[k]

    for t in range(t_len):
        for s in range(t_len):
            if t >= s:
                cross = f_power(t - s) @ var[s]
            else:
                cross = var[t] @ f_power(s - t).T
            sxx[t * state_dim : (t + 1) * state_dim, s * state_dim : (s + 1) * state_dim] = cross

    rows = []
    y_obs = []
    mu_y = []
    for t in range(t_len):
        idx = np.flatnonzero(observed[t])
        if idx.size == 0:
            continue
        for i in idx:
            row = np.zeros(dim, dtype=np.float64)
            row[t * state_dim : (t + 1) * state_dim] = h[i]
            rows.append(row)
            y_obs.append(y[t, i])
            mu_y.append(float(h[i] @ mu[t]))
    design = np.stack(rows, axis=0)
    yv = np.asarray(y_obs, dtype=np.float64)
    muy = np.asarray(mu_y, dtype=np.float64)
    # Observation-noise covariance is block-diagonal across time. Within a
    # time, components that are jointly observed keep their R sub-block.
    sy_noise = np.zeros((len(yv), len(yv)), dtype=np.float64)
    cursor = 0
    for t in range(t_len):
        idx = np.flatnonzero(observed[t])
        n = idx.size
        if n == 0:
            continue
        sy_noise[cursor : cursor + n, cursor : cursor + n] = r[np.ix_(idx, idx)]
        cursor += n
    sy = design @ sxx @ design.T + sy_noise
    sxy = sxx @ design.T
    corr = sxy @ np.linalg.solve(sy, yv - muy)
    return (mu.reshape(-1) + corr).reshape(t_len, state_dim)


def _mask_random(rng, shape, rate, force_keep_ends=True):
    observed = rng.random(shape) > rate
    if force_keep_ends:
        observed[0] = True
        observed[-1] = True
    # Keep at least one observation in every component.
    for i in range(observed.shape[1]):
        if not observed[:, i].any():
            observed[shape[0] // 2, i] = True
    return observed


def test_rts_matches_exact_conditional_mean_under_masking():
    rng = np.random.default_rng(0)
    f = np.array([[0.75, 0.15], [-0.2, 0.55]], dtype=np.float64)
    h = np.eye(2)
    q = np.array([[0.08, 0.01], [0.01, 0.05]], dtype=np.float64)
    r = np.diag([0.15, 0.25])
    mu0 = np.array([0.2, -0.4])
    p0 = np.eye(2) * 0.5
    x, y = simulate_with_initial_cov(f, h, q, r, mu0, p0, t_len=18, rng=rng)
    observed = _mask_random(rng, y.shape, rate=0.35)
    # Drop a whole time and one component of another time.
    observed[5] = False
    observed[9, 1] = False
    y_masked = y.copy()
    y_masked[~observed] = np.nan

    model = LinearGaussianEM(
        state_dim=2,
        obs_dim=2,
        estimate_emission=False,
        standardize=False,
        n_iter=1,
    )
    model.set_parameters(F=f, H=h, Q=q, R=r, mu0=mu0, P0=p0)
    smooth = model.filter_smooth(y_masked, observed)
    exact = _batch_posterior_mean(model, np.where(observed, y, 0.0), observed)
    np.testing.assert_allclose(smooth.smoothed_mean, exact, atol=1e-6, rtol=1e-5)


def test_missing_time_skips_the_update():
    rng = np.random.default_rng(1)
    f = np.array([[0.6]])
    h = np.array([[1.0]])
    q = np.array([[0.2]])
    r = np.array([[0.3]])
    model = LinearGaussianEM(state_dim=1, obs_dim=1, standardize=False)
    model.set_parameters(F=f, H=h, Q=q, R=r, mu0=np.zeros(1), P0=np.eye(1))
    y = rng.normal(size=(6, 1))
    observed = np.ones((6, 1), dtype=bool)
    observed[3] = False
    result = model.filter_smooth(y, observed)
    np.testing.assert_allclose(result.filtered_mean[3], result.predicted_mean[3])
    np.testing.assert_allclose(result.filtered_cov[3], result.predicted_cov[3])
    # A fully observed neighbor is actually updated.
    assert not np.allclose(result.filtered_mean[2], result.predicted_mean[2])


def test_partial_observation_uses_only_the_observed_row():
    f = np.eye(2) * 0.5
    h = np.eye(2)
    q = np.eye(2) * 0.2
    r = np.eye(2) * 0.1
    model = LinearGaussianEM(state_dim=2, obs_dim=2, standardize=False)
    model.set_parameters(
        F=f, H=h, Q=q, R=r, mu0=np.zeros(2), P0=np.eye(2)
    )
    y = np.array([[1.0, 50.0], [0.2, -0.4]])
    observed = np.array([[True, False], [True, True]])
    result = model.filter_smooth(y, observed)
    # At t=0 the prior mean is 0 and only component 0 is seen, so the
    # filtered mean of component 0 moves toward the data and component 1
    # stays at the prior (independent noise, diagonal prior).
    assert result.filtered_mean[0, 0] > 0.5
    np.testing.assert_allclose(result.filtered_mean[0, 1], 0.0, atol=1e-8)


def test_known_parameters_smoother_imputes_masked_observations():
    """Sanity check: with the true system, RTS fills masked entries."""
    rng = np.random.default_rng(4)
    f = np.array([[0.95, 0.02], [-0.04, 0.92]], dtype=np.float64)
    h = np.eye(2)
    q = np.eye(2) * 0.01
    r = np.eye(2) * 0.01
    _x, y = simulate_with_initial_cov(
        f, h, q, r, np.zeros(2), np.eye(2), t_len=250, rng=rng
    )
    observed = _mask_random(rng, y.shape, rate=0.5)
    y_masked = y.copy()
    y_masked[~observed] = np.nan
    model = LinearGaussianEM(
        state_dim=2, obs_dim=2, estimate_emission=False, standardize=False
    )
    model.set_parameters(F=f, H=h, Q=q, R=r, mu0=np.zeros(2), P0=np.eye(2))
    imputed = model.impute(y_masked, observed)
    missing = ~observed
    err_smooth = np.mean((imputed["smoothed"][missing] - y[missing]) ** 2)
    err_filt = np.mean((imputed["filtered"][missing] - y[missing]) ** 2)
    err_zero = np.mean(y[missing] ** 2)
    assert err_smooth < 0.25 * err_zero
    assert err_filt < 0.45 * err_zero
    assert err_smooth < err_filt


def test_em_moves_toward_the_true_transition_under_masking():
    rng = np.random.default_rng(2)
    f = np.array([[0.85, 0.05], [-0.15, 0.7]], dtype=np.float64)
    h = np.eye(2)
    q = np.diag([0.04, 0.06])
    r = np.diag([0.05, 0.08])
    mu0 = np.zeros(2)
    p0 = np.eye(2)
    _x, y = simulate_with_initial_cov(f, h, q, r, mu0, p0, t_len=800, rng=rng)
    observed = _mask_random(rng, y.shape, rate=0.3, force_keep_ends=False)
    y_masked = y.copy()
    y_masked[~observed] = np.nan

    model = LinearGaussianEM(
        state_dim=2,
        obs_dim=2,
        estimate_emission=False,
        r_structure="diagonal",
        standardize=False,
        n_iter=40,
        tol=1e-6,
        min_variance=1e-8,
    )
    model.set_parameters(
        H=h, mu0=mu0, P0=np.eye(2), F=0.2 * np.eye(2), Q=np.eye(2), R=np.eye(2)
    )
    init_err = np.linalg.norm(model.F - f)
    model.fit([y_masked], [observed], init="identity")
    final_err = np.linalg.norm(model.F - f)
    # H is held at I so F, Q, R are identifiable. Finite-sample EM on a masked
    # series will not reproduce F to machine precision; it should get close
    # and beat the initialization.
    assert final_err < init_err
    assert final_err < 0.2
    imputed = model.impute(y_masked, observed)
    missing = ~observed
    err_smooth = np.mean((imputed["smoothed"][missing] - y[missing]) ** 2)
    err_zero = np.mean(y[missing] ** 2)
    assert err_smooth < err_zero


def test_autonomous_suffix_ignores_future_observations():
    rng = np.random.default_rng(3)
    f = np.array([[0.9]])
    h = np.array([[1.0]])
    q = np.array([[0.05]])
    r = np.array([[0.02]])
    _x, y = simulate_with_initial_cov(
        f, h, q, r, np.zeros(1), np.eye(1), t_len=30, rng=rng
    )
    model = LinearGaussianEM(state_dim=1, obs_dim=1, standardize=False, estimate_emission=False)
    model.set_parameters(F=f, H=h, Q=q, R=r, mu0=np.zeros(1), P0=np.eye(1))
    observed = np.ones(y.shape, dtype=bool)
    auto = np.zeros(len(y), dtype=bool)
    auto[15:] = True
    out = model.impute(y, observed, auto_mask=auto)
    blocked = observed.copy()
    blocked[15:] = False
    ref = model.impute(y, blocked)
    np.testing.assert_allclose(out["autonomous"], ref["predicted"])
    # Before the flip the open-loop run still assimilates, so it matches the
    # causal one-step predictions. After the flip it must diverge from a run
    # that keeps seeing data, on this noisy series.
    np.testing.assert_allclose(out["autonomous"][:15], out["predicted"][:15])
    assert np.mean((out["autonomous"][15:] - out["predicted"][15:]) ** 2) > 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
