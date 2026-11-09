"""Oracle EKF / UKF that integrate the Lorenz-63 vector field used by this repo.

This is an upper bound, not a fair competitor to OTF. The discrete step is the
sequential forward Euler in ``dvae.dataset.generate_save_lorenz_63.L63.step``
(x is updated before it is used in the y and z increments). Parameters
(sigma, rho, beta, dt) are inputs. The dataset generator currently writes
``rho=40`` in source, while ``Lorenz63`` defaults the folder name to ``rho=28``.
``parse_lorenz_label`` reads whichever label is on the files being scored.

Process noise and observation noise are small tuning constants. They are not
learned. The simulated trajectories are deterministic, so these variances only
keep the filter from collapsing.

The smoother is the extended RTS recursion (EKF) or the unscented RTS
recursion that uses the sigma-point cross-covariance (UKF). Tresp & Hofmann
discuss the EKF as a filter; they do not write this backward pass. It is
labeled ``oracle`` so it is not confused with the Tresp baseline.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from dvae.baselines.filtering.kalman_em import _psd, _solve_spd, _sym

Array = np.ndarray

ROLE = "oracle"
ORACLE_NOTE = (
    "Oracle EKF/UKF using the true Lorenz-63 equations (sequential Euler "
    "matching L63.step). Not a fair competitor to learned OTF models."
)

_LABEL_RE = re.compile(
    r"sigma(?P<sigma>[0-9.]+)_rho(?P<rho>[0-9.]+)_beta(?P<num>[0-9]+)d(?P<den>[0-9]+)"
    r".*_dt(?P<dt>[0-9.eE+-]+)"
)

OBSERVATION_INDEX = {
    "only_x": (0,),
    "only_xy": (0, 1),
    "only_xz": (0, 2),
    "all_xyz": (0, 1, 2),
    "xyz_to_xyz": (0, 1, 2),
}


def parse_lorenz_label(label: str) -> Optional[Tuple[float, float, float, float]]:
    """Return ``(sigma, rho, beta, dt)`` parsed from a dataset folder name."""
    if not label:
        return None
    match = _LABEL_RE.search(str(label))
    if match is None:
        return None
    beta = float(match.group("num")) / float(match.group("den"))
    return (
        float(match.group("sigma")),
        float(match.group("rho")),
        beta,
        float(match.group("dt")),
    )


def lorenz63_euler_step(
    state: Array, sigma: float, rho: float, beta: float, dt: float
) -> Array:
    """One sequential Euler step, matching ``L63.step``."""
    x, y, z = (float(v) for v in np.asarray(state, dtype=np.float64).reshape(-1))
    x = x + (sigma * (y - x)) * dt
    y = y + (x * (rho - z)) * dt
    z = z + (x * y - beta * z) * dt
    return np.array([x, y, z], dtype=np.float64)


def lorenz63_euler_jacobian(
    state: Array, sigma: float, rho: float, beta: float, dt: float
) -> Array:
    """Analytic Jacobian of ``lorenz63_euler_step`` at ``state``."""
    x, y, z = (float(v) for v in np.asarray(state, dtype=np.float64).reshape(-1))
    dx1_dx = 1.0 - sigma * dt
    dx1_dy = sigma * dt
    dx1_dz = 0.0
    x1 = x + (sigma * (y - x)) * dt
    a = (rho - z) * dt
    y1 = y + x1 * a
    dy1_dx = a * dx1_dx
    dy1_dy = 1.0 + a * dx1_dy
    dy1_dz = -x1 * dt
    dz1_dx = dt * (y1 * dx1_dx + x1 * dy1_dx)
    dz1_dy = dt * (y1 * dx1_dy + x1 * dy1_dy)
    dz1_dz = (1.0 - beta * dt) + dt * (y1 * dx1_dz + x1 * dy1_dz)
    return np.array(
        [
            [dx1_dx, dx1_dy, dx1_dz],
            [dy1_dx, dy1_dy, dy1_dz],
            [dz1_dx, dz1_dy, dz1_dz],
        ],
        dtype=np.float64,
    )


def observation_matrix(observation_process: str) -> Array:
    if observation_process not in OBSERVATION_INDEX:
        known = ", ".join(sorted(OBSERVATION_INDEX))
        raise ValueError(
            f"Oracle observation {observation_process!r} is not one of: {known}. "
            "Noise and indicate/interpolate variants are not an oracle observation; "
            "pass the underlying slice (only_x, only_xy, only_xz, all_xyz)."
        )
    idx = OBSERVATION_INDEX[observation_process]
    h = np.zeros((len(idx), 3), dtype=np.float64)
    for row, col in enumerate(idx):
        h[row, col] = 1.0
    return h


def _sigma_points(mean: Array, cov: Array, alpha: float, beta_ut: float, kappa: float):
    n = mean.shape[0]
    lam = alpha**2 * (n + kappa) - n
    scale = n + lam
    chol_cov = _psd(cov, eps=1e-10) * scale
    try:
        chol = np.linalg.cholesky(chol_cov)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(chol_cov + 1e-6 * np.eye(n))
    pts = [mean]
    for i in range(n):
        pts.append(mean + chol[:, i])
        pts.append(mean - chol[:, i])
    pts_arr = np.stack(pts, axis=0)
    wm = np.full(2 * n + 1, 0.5 / scale, dtype=np.float64)
    wc = wm.copy()
    wm[0] = lam / scale
    wc[0] = lam / scale + (1.0 - alpha**2 + beta_ut)
    return pts_arr, wm, wc


class LorenzOracleFilter:
    """Causal filter and RTS smoother in the true 3-D Lorenz state."""

    def __init__(
        self,
        observation_process: str,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        dt: float = 1e-2,
        kind: str = "ekf",
        q_var: float = 1e-4,
        r_var: float = 1e-3,
        p0_var: float = 10.0,
        ukf_alpha: float = 1e-3,
        ukf_beta: float = 2.0,
        ukf_kappa: float = 0.0,
        cov_clip: float = 1e6,
    ) -> None:
        if kind not in ("ekf", "ukf"):
            raise ValueError("kind must be 'ekf' or 'ukf'")
        self.kind = kind
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)
        self.dt = float(dt)
        self.H = observation_matrix(observation_process)
        self.observation_process = observation_process
        self.obs_dim = int(self.H.shape[0])
        self.Q = np.eye(3, dtype=np.float64) * float(q_var)
        self.R = np.eye(self.obs_dim, dtype=np.float64) * float(r_var)
        self.p0_var = float(p0_var)
        self.ukf_alpha = float(ukf_alpha)
        self.ukf_beta = float(ukf_beta)
        self.ukf_kappa = float(ukf_kappa)
        self.cov_clip = float(cov_clip)
        self.role = ROLE

    def _step(self, state: Array) -> Array:
        return lorenz63_euler_step(state, self.sigma, self.rho, self.beta, self.dt)

    def _jacobian(self, state: Array) -> Array:
        return lorenz63_euler_jacobian(
            state, self.sigma, self.rho, self.beta, self.dt
        )

    def _clip_cov(self, cov: Array) -> Array:
        cov = _psd(cov, eps=1e-12)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-12, self.cov_clip)
        return (vecs * vals) @ vecs.T

    def _initial(self, y: Array, observed: Array) -> Tuple[Array, Array]:
        mu = np.zeros(3, dtype=np.float64)
        for t in range(y.shape[0]):
            idx = np.flatnonzero(observed[t])
            if idx.size == 0:
                continue
            # H[i] picks a standard basis vector.
            for i in idx:
                coord = int(np.flatnonzero(self.H[i])[0])
                mu[coord] = y[t, i]
            break
        p0 = np.eye(3, dtype=np.float64) * self.p0_var
        return mu, p0

    def _predict(self, mean: Array, cov: Array) -> Tuple[Array, Array, Array]:
        """Return predicted mean, covariance, and cross-cov Cov(x_prev, x_next)."""
        if self.kind == "ekf":
            jac = self._jacobian(mean)
            pred = self._step(mean)
            p_pred = self._clip_cov(jac @ cov @ jac.T + self.Q)
            cross = cov @ jac.T
            return pred, p_pred, cross
        pts, wm, wc = _sigma_points(
            mean, cov, self.ukf_alpha, self.ukf_beta, self.ukf_kappa
        )
        prop = np.stack([self._step(p) for p in pts], axis=0)
        pred = wm @ prop
        diff = prop - pred
        p_pred = self.Q.copy()
        cross = np.zeros((3, 3), dtype=np.float64)
        centered_in = pts - mean
        for i in range(pts.shape[0]):
            p_pred += wc[i] * np.outer(diff[i], diff[i])
            cross += wc[i] * np.outer(centered_in[i], diff[i])
        return pred, self._clip_cov(p_pred), cross

    def _update(
        self, mean: Array, cov: Array, yt: Array, idx: Array
    ) -> Tuple[Array, Array]:
        if idx.size == 0:
            return mean, cov
        ht = self.H[idx]
        rt = self.R[np.ix_(idx, idx)]
        innov_cov = _sym(ht @ cov @ ht.T + rt)
        gain = _solve_spd(innov_cov, ht @ cov).T
        innov = yt - ht @ mean
        new_mean = mean + gain @ innov
        eye = np.eye(3, dtype=np.float64)
        ikh = eye - gain @ ht
        new_cov = self._clip_cov(ikh @ cov @ ikh.T + gain @ rt @ gain.T)
        return new_mean, new_cov

    def filter_smooth(
        self, y: Array, observed: Optional[Array] = None
    ) -> Dict[str, Array]:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape[1] != self.obs_dim:
            raise ValueError(
                f"{self.observation_process} expects obs dim {self.obs_dim}, "
                f"got {y.shape[1]}"
            )
        if observed is None:
            observed = np.isfinite(y)
        else:
            observed = np.asarray(observed, dtype=bool)
            if observed.ndim == 1:
                observed = observed.reshape(-1, 1)
        observed = observed & np.isfinite(y)
        t_len = y.shape[0]
        x_pred = np.zeros((t_len, 3), dtype=np.float64)
        p_pred = np.zeros((t_len, 3, 3), dtype=np.float64)
        x_filt = np.zeros((t_len, 3), dtype=np.float64)
        p_filt = np.zeros((t_len, 3, 3), dtype=np.float64)
        cross = np.zeros((t_len, 3, 3), dtype=np.float64)
        mu, cov = self._initial(y, observed)
        for t in range(t_len):
            if t == 0:
                xp, pp, _cross = mu, cov, np.zeros((3, 3))
            else:
                xp, pp, cross_t = self._predict(x_filt[t - 1], p_filt[t - 1])
                cross[t] = cross_t
            x_pred[t] = xp
            p_pred[t] = pp
            idx = np.flatnonzero(observed[t])
            yt = y[t, idx] if idx.size else y[t, :0]
            xf, pf = self._update(xp, pp, yt, idx)
            x_filt[t] = xf
            p_filt[t] = pf

        x_s = x_filt.copy()
        p_s = p_filt.copy()
        for t in range(t_len - 2, -1, -1):
            # cross[t+1] estimates Cov(x_t, x_{t+1} | prediction), so
            # J_t = Cov(x_t, x_{t+1}) Cov(x_{t+1})^{-1}.
            j = _solve_spd(p_pred[t + 1], cross[t + 1].T).T
            x_s[t] = x_filt[t] + j @ (x_s[t + 1] - x_pred[t + 1])
            p_s[t] = self._clip_cov(
                p_filt[t] + j @ (p_s[t + 1] - p_pred[t + 1]) @ j.T
            )
        return {
            "predicted_latent": x_pred,
            "filtered_latent": x_filt,
            "smoothed_latent": x_s,
            "predicted_cov": p_pred,
            "filtered_cov": p_filt,
            "smoothed_cov": p_s,
        }

    def _to_obs(self, latent: Array) -> Array:
        return latent @ self.H.T

    def impute(
        self,
        y: Array,
        observed: Optional[Array] = None,
        *,
        auto_mask: Optional[Array] = None,
    ) -> Dict[str, Array]:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if observed is None:
            observed_b = np.isfinite(y)
        else:
            observed_b = np.asarray(observed, dtype=bool)
            if observed_b.ndim == 1:
                observed_b = observed_b.reshape(-1, 1)
            observed_b = observed_b & np.isfinite(y)
        causal = self.filter_smooth(y, observed_b)
        out = {
            "predicted": self._to_obs(causal["predicted_latent"]),
            "filtered": self._to_obs(causal["filtered_latent"]),
            "smoothed": self._to_obs(causal["smoothed_latent"]),
            "latent_predicted": causal["predicted_latent"],
            "latent_filtered": causal["filtered_latent"],
            "latent_smoothed": causal["smoothed_latent"],
            "role": self.role,
            "kind": self.kind,
            "note": ORACLE_NOTE,
        }
        if auto_mask is not None:
            auto_mask = np.asarray(auto_mask, dtype=bool).reshape(-1)
            blocked = observed_b & ~auto_mask.reshape(-1, 1)
            rolled = self.filter_smooth(y, blocked)
            out["autonomous"] = self._to_obs(rolled["predicted_latent"])
            out["latent_autonomous"] = rolled["predicted_latent"]
        return out


def integrate_lorenz(
    n_steps: int,
    sigma: float,
    rho: float,
    beta: float,
    dt: float,
    init: Sequence[float],
) -> Array:
    """Deterministic trajectory, shape (n_steps, 3), including the initial state."""
    state = np.asarray(init, dtype=np.float64).reshape(3)
    hist = [state.copy()]
    for _ in range(n_steps - 1):
        state = lorenz63_euler_step(state, sigma, rho, beta, dt)
        hist.append(state.copy())
    return np.stack(hist, axis=0)
