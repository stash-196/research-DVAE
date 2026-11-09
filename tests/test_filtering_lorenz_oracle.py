"""Oracle Lorenz EKF/UKF. Synthetic trajectories only."""

from __future__ import annotations

import numpy as np

from dvae.baselines.filtering.lorenz_oracle import (
    ROLE,
    LorenzOracleFilter,
    integrate_lorenz,
    lorenz63_euler_jacobian,
    lorenz63_euler_step,
    parse_lorenz_label,
)


def test_step_matches_sequential_euler_and_jacobian_matches_differences():
    state = np.array([1.0, 2.0, 3.0])
    sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01
    x, y, z = state
    x = x + (sigma * (y - x)) * dt
    y = y + (x * (rho - z)) * dt
    z = z + (x * y - beta * z) * dt
    np.testing.assert_allclose(
        lorenz63_euler_step(state, sigma, rho, beta, dt),
        np.array([x, y, z]),
    )
    jac = lorenz63_euler_jacobian(state, sigma, rho, beta, dt)
    eps = 1e-6
    numerical = np.zeros((3, 3))
    base = lorenz63_euler_step(state, sigma, rho, beta, dt)
    for i in range(3):
        bumped = state.copy()
        bumped[i] += eps
        numerical[:, i] = (
            lorenz63_euler_step(bumped, sigma, rho, beta, dt) - base
        ) / eps
    np.testing.assert_allclose(jac, numerical, atol=1e-5)


def test_label_parser_reads_rho_and_beta_fraction():
    parsed = parse_lorenz_label("sigma10_rho28_beta8d3_N108k_dt0.01")
    assert parsed is not None
    sigma, rho, beta, dt = parsed
    assert sigma == 10.0
    assert rho == 28.0
    np.testing.assert_allclose(beta, 8.0 / 3.0)
    assert dt == 0.01
    assert parse_lorenz_label("not_a_label") is None


def test_full_observation_ekf_tracks_the_true_trajectory():
    sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01
    xyz = integrate_lorenz(80, sigma, rho, beta, dt, init=(1.0, 1.0, 1.0))
    filt = LorenzOracleFilter(
        "all_xyz",
        sigma=sigma,
        rho=rho,
        beta=beta,
        dt=dt,
        kind="ekf",
        q_var=1e-8,
        r_var=1e-6,
        p0_var=1e-6,
    )
    out = filt.impute(xyz, np.ones(xyz.shape, dtype=bool))
    # Started on the true state with a tiny prior, the oracle should stay close.
    err = np.mean((out["filtered"] - xyz) ** 2)
    assert err < 1e-4
    assert out["role"] == ROLE
    assert "Not a fair competitor" in out["note"]


def test_only_x_mask_reports_filtered_and_smoothed_series():
    sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01
    xyz = integrate_lorenz(60, sigma, rho, beta, dt, init=(1.0, 1.0, 1.0))
    y = xyz[:, :1]
    missing = np.zeros(y.shape, dtype=bool)
    missing[10:20] = True
    observed = ~missing
    filt = LorenzOracleFilter(
        "only_x",
        sigma=sigma,
        rho=rho,
        beta=beta,
        dt=dt,
        kind="ukf",
        q_var=1e-4,
        r_var=1e-3,
        p0_var=1.0,
    )
    out = filt.impute(np.where(missing, np.nan, y), observed)
    assert out["filtered"].shape == y.shape
    assert out["smoothed"].shape == y.shape
    assert np.isfinite(out["filtered"]).all()
    assert np.isfinite(out["smoothed"]).all()
    # On a gap, the smoother (which sees both sides) should beat a pure
    # forecast that never saw x, measured against the true x.
    gap = missing[:, 0]
    smooth_err = np.mean((out["smoothed"][gap] - y[gap]) ** 2)
    assert smooth_err < np.mean(y[gap] ** 2)
