"""Linear-Gaussian state-space imputation with missing observations.

Formulation
-----------
State-space model, Shumway & Stoffer (1982) indexing::

    x_0 ~ N(mu0, P0)
    x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q),  t = 1..T-1
    y_t = H x_t + v_t,       v_t ~ N(0, R)

Missing data follows Shumway & Stoffer (1982): if nothing is observed at
time t the Kalman gain is zero and the filter keeps the predicted state.
Component-wise gaps (a row of y present, another absent) are the textbook
extension: the update uses only the observed rows of H and R
(Shumway & Stoffer, *Time Series Analysis and Its Applications*, missing-data
modifications). The 1982 paper itself treats a time index as entirely
present or entirely missing.

The RTS smoother and the lag-one covariance recursion are the ones used by
the EM M-step in Shumway & Stoffer (1982). With ``estimate_emission=True``
and a diagonal R, each observed component has its own closed-form regression
M-step. A full (non-diagonal) R is updated only on time indices where the
whole observation vector is present; partial times still enter the state
posterior and therefore F and Q. That restriction is stated because a full
R couples missing components into the complete-data likelihood, and the 1982
paper does not give a partial-vector M-step.

Parameters are identifiable for the recovery test when H is held at I.
If H is free and the state dimension equals the observation dimension, F is
only identified up to a linear change of basis. Optional per-component
standardization is a numerical device; under H = I it changes F to
D^{-1} F D when component scales differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


def _sym(a: Array) -> Array:
    return 0.5 * (a + a.T)


def _psd(a: Array, eps: float = 1e-8) -> Array:
    a = _sym(np.asarray(a, dtype=np.float64))
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T


def _solve_spd(a: Array, b: Array, jitter: float = 1e-9) -> Array:
    a = _sym(np.asarray(a, dtype=np.float64))
    eye = np.eye(a.shape[0], dtype=np.float64)
    return np.linalg.solve(a + jitter * eye, b)


def _as_series(y: Array) -> Array:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError(f"Expected y with shape (T,) or (T, D), got {y.shape}")
    return y


def _as_observed(observed: Optional[Array], y: Array) -> Array:
    if observed is None:
        return np.isfinite(y)
    observed = np.asarray(observed, dtype=bool)
    if observed.ndim == 1:
        if y.shape[1] != 1:
            raise ValueError("1-D observation mask requires a univariate series")
        observed = observed.reshape(-1, 1)
    if observed.shape != y.shape:
        raise ValueError(
            f"observed shape {observed.shape} does not match y shape {y.shape}"
        )
    return observed & np.isfinite(y)


@dataclass
class FilterSmoothResult:
    """Latent trajectories. ``predicted`` is x_{t|t-1} (x_{0|-1} = prior)."""

    predicted_mean: Array
    predicted_cov: Array
    filtered_mean: Array
    filtered_cov: Array
    smoothed_mean: Array
    smoothed_cov: Array
    lag_cov: Array
    loglik: float


def _kalman_filter(
    y: Array,
    observed: Array,
    F: Array,
    H: Array,
    Q: Array,
    R: Array,
    mu0: Array,
    P0: Array,
) -> Tuple[Array, Array, Array, Array, List[Optional[Array]], List[Optional[Array]], float]:
    t_len, _ = y.shape
    state_dim = F.shape[0]
    eye = np.eye(state_dim, dtype=np.float64)
    x_pred = np.zeros((t_len, state_dim), dtype=np.float64)
    p_pred = np.zeros((t_len, state_dim, state_dim), dtype=np.float64)
    x_filt = np.zeros((t_len, state_dim), dtype=np.float64)
    p_filt = np.zeros((t_len, state_dim, state_dim), dtype=np.float64)
    gains: List[Optional[Array]] = [None] * t_len
    h_rows: List[Optional[Array]] = [None] * t_len
    loglik = 0.0
    two_pi = np.log(2.0 * np.pi)

    x_prev = np.asarray(mu0, dtype=np.float64).reshape(-1)
    p_prev = _psd(P0, eps=1e-10)

    for t in range(t_len):
        if t == 0:
            xp = x_prev.copy()
            pp = p_prev.copy()
        else:
            xp = F @ x_filt[t - 1]
            pp = _sym(F @ p_filt[t - 1] @ F.T + Q)
        x_pred[t] = xp
        p_pred[t] = pp

        idx = np.flatnonzero(observed[t])
        if idx.size == 0:
            x_filt[t] = xp
            p_filt[t] = pp
            continue

        ht = H[idx]
        rt = R[np.ix_(idx, idx)]
        yt = y[t, idx]
        innov_cov = _sym(ht @ pp @ ht.T + rt)
        innov = yt - ht @ xp
        gain = _solve_spd(innov_cov, ht @ pp).T
        x_filt[t] = xp + gain @ innov
        ikh = eye - gain @ ht
        p_filt[t] = _psd(ikh @ pp @ ikh.T + gain @ rt @ gain.T, eps=1e-12)
        gains[t] = gain
        h_rows[t] = ht

        sign, logdet = np.linalg.slogdet(innov_cov + 1e-12 * np.eye(idx.size))
        if sign <= 0:
            logdet = np.log(np.linalg.det(innov_cov + 1e-8 * np.eye(idx.size)) + 1e-30)
        quad = float(innov @ _solve_spd(innov_cov, innov))
        loglik += -0.5 * (logdet + quad + idx.size * two_pi)

    return x_pred, p_pred, x_filt, p_filt, gains, h_rows, float(loglik)


def _rts_smooth(
    F: Array,
    x_pred: Array,
    p_pred: Array,
    x_filt: Array,
    p_filt: Array,
    gains: Sequence[Optional[Array]],
    h_rows: Sequence[Optional[Array]],
) -> Tuple[Array, Array, Array]:
    t_len, state_dim = x_filt.shape
    x_s = x_filt.copy()
    p_s = p_filt.copy()
    cross = np.zeros((t_len, state_dim, state_dim), dtype=np.float64)
    # J_t satisfies E[x_t | x_{t+1}, y] = const + J_t x_{t+1}, so
    # Cov(x_{t}, x_{t-1} | y) = P_{t|T} J_{t-1}^T.
    # ``gains`` and ``h_rows`` are part of the filter return and are not
    # required once filtered means and covariances are known.
    del gains, h_rows
    js: List[Optional[Array]] = [None] * t_len

    for t in range(t_len - 2, -1, -1):
        j = _solve_spd(p_pred[t + 1], F @ p_filt[t]).T
        js[t] = j
        x_s[t] = x_filt[t] + j @ (x_s[t + 1] - x_pred[t + 1])
        p_s[t] = _psd(
            p_filt[t] + j @ (p_s[t + 1] - p_pred[t + 1]) @ j.T,
            eps=1e-12,
        )

    for t in range(1, t_len):
        j_prev = js[t - 1]
        if j_prev is None:
            continue
        cross[t] = p_s[t] @ j_prev.T
    return x_s, p_s, cross


@dataclass
class LinearGaussianEM:
    """EM for one or more series that share (F, H, Q, R, mu0, P0)."""

    state_dim: int
    obs_dim: int
    estimate_emission: bool = True
    r_structure: str = "diagonal"
    standardize: bool = True
    n_iter: int = 25
    tol: float = 1e-4
    seed: int = 0
    min_variance: float = 1e-6

    def __post_init__(self) -> None:
        if self.r_structure not in ("diagonal", "full"):
            raise ValueError("r_structure must be 'diagonal' or 'full'")
        if self.state_dim < 1 or self.obs_dim < 1:
            raise ValueError("state_dim and obs_dim must be positive")
        rng = np.random.default_rng(self.seed)
        self.obs_mean = np.zeros(self.obs_dim, dtype=np.float64)
        self.obs_std = np.ones(self.obs_dim, dtype=np.float64)
        self.mu0 = np.zeros(self.state_dim, dtype=np.float64)
        self.P0 = np.eye(self.state_dim, dtype=np.float64)
        self.F = 0.5 * np.eye(self.state_dim, dtype=np.float64)
        self.Q = 0.1 * np.eye(self.state_dim, dtype=np.float64)
        if self.state_dim == self.obs_dim:
            self.H = np.eye(self.obs_dim, dtype=np.float64)
        else:
            self.H = np.zeros((self.obs_dim, self.state_dim), dtype=np.float64)
            rank = min(self.obs_dim, self.state_dim)
            self.H[:rank, :rank] = np.eye(rank)
        if not self.estimate_emission:
            pass
        self.R = 0.1 * np.eye(self.obs_dim, dtype=np.float64)
        self.loglik_history: List[float] = []
        self._rng = rng

    def set_parameters(
        self,
        *,
        F: Optional[Array] = None,
        H: Optional[Array] = None,
        Q: Optional[Array] = None,
        R: Optional[Array] = None,
        mu0: Optional[Array] = None,
        P0: Optional[Array] = None,
    ) -> None:
        if F is not None:
            self.F = np.asarray(F, dtype=np.float64).copy()
        if H is not None:
            self.H = np.asarray(H, dtype=np.float64).copy()
        if Q is not None:
            self.Q = _psd(Q, eps=self.min_variance)
        if R is not None:
            self.R = _psd(R, eps=self.min_variance)
        if mu0 is not None:
            self.mu0 = np.asarray(mu0, dtype=np.float64).reshape(-1).copy()
        if P0 is not None:
            self.P0 = _psd(P0, eps=self.min_variance)

    def _scale_series(self, y: Array) -> Array:
        return (y - self.obs_mean) / self.obs_std

    def _fit_scale(self, series: Sequence[Array]) -> None:
        if not self.standardize:
            self.obs_mean = np.zeros(self.obs_dim, dtype=np.float64)
            self.obs_std = np.ones(self.obs_dim, dtype=np.float64)
            return
        chunks = []
        for y in series:
            y = _as_series(y)
            flat = y.reshape(-1, self.obs_dim)
            chunks.append(flat[np.isfinite(flat).all(axis=1)])
            # also keep component-wise finite values below
        cols = [[] for _ in range(self.obs_dim)]
        for y in series:
            y = _as_series(y)
            for d in range(self.obs_dim):
                col = y[:, d]
                cols[d].append(col[np.isfinite(col)])
        mean = np.zeros(self.obs_dim, dtype=np.float64)
        std = np.ones(self.obs_dim, dtype=np.float64)
        for d in range(self.obs_dim):
            if cols[d]:
                vals = np.concatenate(cols[d])
                if vals.size:
                    mean[d] = float(np.mean(vals))
                    std[d] = float(np.std(vals))
        std = np.where(std < 1e-8, 1.0, std)
        self.obs_mean = mean
        self.obs_std = std
        del chunks

    def _init_from_ols(self, series: Sequence[Tuple[Array, Array]]) -> None:
        """Lag-1 least squares on linearly filled series, used only as an init."""
        if self.state_dim != self.obs_dim:
            return
        acc_xx = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        acc_yx = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        for y, _observed in series:
            filled = y.copy()
            for d in range(filled.shape[1]):
                col = filled[:, d]
                good = np.isfinite(col)
                if good.any() and (~good).any():
                    idx = np.arange(col.shape[0])
                    col = col.copy()
                    col[~good] = np.interp(idx[~good], idx[good], col[good])
                    filled[:, d] = col
                elif not good.any():
                    filled[:, d] = 0.0
            x0 = filled[:-1]
            x1 = filled[1:]
            acc_xx += x0.T @ x0
            acc_yx += x1.T @ x0
        self.F = _solve_spd(acc_xx, acc_yx.T).T
        resid = []
        for y, _observed in series:
            filled = np.nan_to_num(y, nan=0.0)
            if len(filled) < 2:
                continue
            err = filled[1:] - filled[:-1] @ self.F.T
            resid.append(err)
        if resid:
            err = np.concatenate(resid, axis=0)
            var = np.maximum(np.var(err, axis=0).mean(), self.min_variance)
            self.Q = var * np.eye(self.state_dim)
            self.R = var * np.eye(self.obs_dim)

    def filter_smooth(self, y: Array, observed: Optional[Array] = None) -> FilterSmoothResult:
        y_raw = _as_series(y)
        if y_raw.shape[1] != self.obs_dim:
            raise ValueError(
                f"Expected obs_dim={self.obs_dim}, got {y_raw.shape[1]}"
            )
        observed_b = _as_observed(observed, y_raw)
        y_s = self._scale_series(y_raw)
        y_s = np.where(observed_b, y_s, 0.0)
        x_pred, p_pred, x_filt, p_filt, gains, h_rows, loglik = _kalman_filter(
            y_s, observed_b, self.F, self.H, self.Q, self.R, self.mu0, self.P0
        )
        x_s, p_s, cross = _rts_smooth(
            self.F, x_pred, p_pred, x_filt, p_filt, gains, h_rows
        )
        return FilterSmoothResult(
            predicted_mean=x_pred,
            predicted_cov=p_pred,
            filtered_mean=x_filt,
            filtered_cov=p_filt,
            smoothed_mean=x_s,
            smoothed_cov=p_s,
            lag_cov=cross,
            loglik=loglik,
        )

    def _observation_mean(self, latent: Array) -> Array:
        std = self.obs_std.reshape(1, -1)
        mean = self.obs_mean.reshape(1, -1)
        return (latent @ self.H.T) * std + mean

    def impute(
        self,
        y: Array,
        observed: Optional[Array] = None,
        *,
        auto_mask: Optional[Array] = None,
    ) -> Dict[str, Array]:
        """Causal prediction, filtered posterior, smoother, and open-loop forecast.

        ``predicted`` is H x_{t|t-1} with an update wherever ``observed`` is
        true. That is the one-step output comparable to an OTF teacher-forced
        step (the network emits its prediction, then consumes the observation
        if it exists).

        ``filtered`` is H x_{t|t}. On a fully missing time it equals
        ``predicted``. On a partial time the missing components use the
        instantaneous update from the observed components.

        ``smoothed`` is H x_{t|T} from the RTS smoother (acausal).

        ``autonomous`` is H x_{t|t-1} when updates are suppressed on
        ``auto_mask`` (True = do not assimilate). Pass a half-series mask to
        match an OTF half-and-half free run.
        """
        y_raw = _as_series(y)
        observed_b = _as_observed(observed, y_raw)
        causal = self.filter_smooth(y_raw, observed_b)
        out = {
            "predicted": self._observation_mean(causal.predicted_mean),
            "filtered": self._observation_mean(causal.filtered_mean),
            "smoothed": self._observation_mean(causal.smoothed_mean),
            "latent_filtered": causal.filtered_mean,
            "latent_smoothed": causal.smoothed_mean,
            "loglik": causal.loglik,
        }
        if auto_mask is not None:
            auto_mask = np.asarray(auto_mask, dtype=bool).reshape(-1)
            if auto_mask.shape[0] != y_raw.shape[0]:
                raise ValueError("auto_mask length does not match the series")
            blocked = observed_b & ~auto_mask.reshape(-1, 1)
            rolled = self.filter_smooth(y_raw, blocked)
            out["autonomous"] = self._observation_mean(rolled.predicted_mean)
            out["latent_autonomous"] = rolled.predicted_mean
        return out

    def fit(
        self,
        series: Sequence[Array],
        observed: Optional[Sequence[Optional[Array]]] = None,
        *,
        init: str = "identity",
    ) -> "LinearGaussianEM":
        """Run EM. ``series`` entries are (T_i, obs_dim) arrays, NaN = missing."""
        if observed is None:
            observed_list: List[Optional[Array]] = [None] * len(series)
        else:
            observed_list = list(observed)
        if len(observed_list) != len(series):
            raise ValueError("observed and series must have the same length")
        prepared: List[Tuple[Array, Array]] = []
        for y, obs in zip(series, observed_list):
            y_arr = _as_series(y)
            if y_arr.shape[1] != self.obs_dim:
                raise ValueError(
                    f"Expected obs_dim={self.obs_dim}, got {y_arr.shape[1]}"
                )
            prepared.append((y_arr, _as_observed(obs, y_arr)))
        self._fit_scale([y for y, _ in prepared])
        scaled = [(self._scale_series(y), obs) for y, obs in prepared]
        if init == "ols":
            self._init_from_ols(scaled)
        elif init != "identity":
            raise ValueError("init must be 'identity' or 'ols'")

        any_partial = False
        for _y, obs in scaled:
            row_any = obs.any(axis=1)
            row_all = obs.all(axis=1)
            if np.any(row_any & ~row_all):
                any_partial = True
                break
        r_structure = self.r_structure
        if any_partial and r_structure == "full":
            # Full-R M-step below ignores partial times. Keep the requested
            # structure, but the filter still assimilates those times.
            pass

        prev_ll = -np.inf
        self.loglik_history = []
        for _it in range(self.n_iter):
            stats = self._accumulate(scaled)
            self._mstep(stats)
            ll = stats["loglik"]
            self.loglik_history.append(ll)
            if np.isfinite(prev_ll) and abs(ll - prev_ll) < self.tol * (
                1.0 + abs(prev_ll)
            ):
                break
            prev_ll = ll
        return self

    def _accumulate(self, scaled: Sequence[Tuple[Array, Array]]) -> Dict[str, Array]:
        d_s = self.state_dim
        d_o = self.obs_dim
        s11 = np.zeros((d_s, d_s), dtype=np.float64)
        s10 = np.zeros((d_s, d_s), dtype=np.float64)
        s00 = np.zeros((d_s, d_s), dtype=np.float64)
        n_trans = 0
        comp_w = [np.zeros((d_s, d_s), dtype=np.float64) for _ in range(d_o)]
        comp_b = [np.zeros(d_s, dtype=np.float64) for _ in range(d_o)]
        comp_n = np.zeros(d_o, dtype=np.float64)
        full_w = np.zeros((d_s, d_s), dtype=np.float64)
        full_b = np.zeros((d_o, d_s), dtype=np.float64)
        full_n = 0
        mu_acc = np.zeros(d_s, dtype=np.float64)
        p_acc = np.zeros((d_s, d_s), dtype=np.float64)
        n_series = 0
        loglik = 0.0
        # E-step trajectories. The M-step for R must use these moments, not a
        # second smooth under the updated H.
        stored = []

        for y, obs in scaled:
            result = self._filter_scaled(y, obs)
            loglik += result.loglik
            t_len = y.shape[0]
            xs = result.smoothed_mean
            ps = result.smoothed_cov
            lag = result.lag_cov
            stored.append((y, obs, xs, ps))
            if t_len >= 1:
                mu_acc += xs[0]
                p_acc += ps[0]
                n_series += 1
            for t in range(1, t_len):
                x_prev = xs[t - 1]
                x_cur = xs[t]
                s11 += ps[t - 1] + np.outer(x_prev, x_prev)
                s10 += lag[t] + np.outer(x_cur, x_prev)
                s00 += ps[t] + np.outer(x_cur, x_cur)
                n_trans += 1
            for t in range(t_len):
                second = ps[t] + np.outer(xs[t], xs[t])
                if self.r_structure == "full" and bool(obs[t].all()):
                    full_w += second
                    full_b += np.outer(y[t], xs[t])
                    full_n += 1
                for i in range(d_o):
                    if not obs[t, i]:
                        continue
                    comp_w[i] += second
                    comp_b[i] += y[t, i] * xs[t]
                    comp_n[i] += 1.0

        return {
            "s11": s11,
            "s10": s10,
            "s00": s00,
            "n_trans": n_trans,
            "comp_w": comp_w,
            "comp_b": comp_b,
            "comp_n": comp_n,
            "full_w": full_w,
            "full_b": full_b,
            "full_n": full_n,
            "mu_acc": mu_acc,
            "p_acc": p_acc,
            "n_series": n_series,
            "loglik": loglik,
            "stored": stored,
        }

    def _filter_scaled(self, y: Array, observed: Array) -> FilterSmoothResult:
        y_filled = np.where(observed, y, 0.0)
        x_pred, p_pred, x_filt, p_filt, gains, h_rows, loglik = _kalman_filter(
            y_filled, observed, self.F, self.H, self.Q, self.R, self.mu0, self.P0
        )
        x_s, p_s, cross = _rts_smooth(
            self.F, x_pred, p_pred, x_filt, p_filt, gains, h_rows
        )
        return FilterSmoothResult(
            x_pred, p_pred, x_filt, p_filt, x_s, p_s, cross, loglik
        )

    def _mstep(self, stats: Dict[str, Array]) -> None:
        n_trans = int(stats["n_trans"])
        if n_trans > 0:
            s11 = stats["s11"]
            s10 = stats["s10"]
            s00 = stats["s00"]
            self.F = _solve_spd(s11, s10.T).T
            q = (s00 - self.F @ s10.T) / n_trans
            self.Q = _psd(q, eps=self.min_variance)
        n_series = int(stats["n_series"])
        if n_series > 0:
            self.mu0 = stats["mu_acc"] / n_series
            # Average of the smoothed marginals. For one series this is P_{0|T}.
            self.P0 = _psd(stats["p_acc"] / n_series, eps=self.min_variance)

        if self.r_structure == "diagonal" or int(stats["full_n"]) == 0:
            h = np.zeros_like(self.H)
            r_diag = np.zeros(self.obs_dim, dtype=np.float64)
            for i in range(self.obs_dim):
                n_i = float(stats["comp_n"][i])
                if n_i < 1:
                    h[i] = self.H[i]
                    r_diag[i] = max(float(self.R[i, i]), self.min_variance)
                    continue
                w = stats["comp_w"][i]
                b = stats["comp_b"][i]
                if self.estimate_emission:
                    h[i] = _solve_spd(w, b)
                else:
                    h[i] = self.H[i]
            if self.estimate_emission:
                self.H = h
            # E[(y - Hx)^2] = (y - H x_{t|T})^2 + H P_{t|T} H^T, with the
            # E-step posterior and the updated H (Shumway & Stoffer M-step).
            for i in range(self.obs_dim):
                n_i = float(stats["comp_n"][i])
                if n_i < 1:
                    continue
                sse = 0.0
                for y, obs, xs, ps in stats["stored"]:
                    for t in range(y.shape[0]):
                        if not obs[t, i]:
                            continue
                        mean = float(self.H[i] @ xs[t])
                        var = float(self.H[i] @ ps[t] @ self.H[i])
                        sse += (y[t, i] - mean) ** 2 + var
                r_diag[i] = max(sse / n_i, self.min_variance)
            self.R = np.diag(r_diag)
            return

        full_n = int(stats["full_n"])
        if self.estimate_emission:
            self.H = _solve_spd(stats["full_w"], stats["full_b"].T).T
        sse = np.zeros((self.obs_dim, self.obs_dim), dtype=np.float64)
        counted = 0
        for y, obs, xs, ps in stats["stored"]:
            for t in range(y.shape[0]):
                if not bool(obs[t].all()):
                    continue
                resid = y[t] - self.H @ xs[t]
                sse += np.outer(resid, resid) + self.H @ ps[t] @ self.H.T
                counted += 1
        if counted == 0:
            counted = full_n
        self.R = _psd(sse / max(counted, 1), eps=self.min_variance)


def simulate_with_initial_cov(
    F: Array,
    H: Array,
    Q: Array,
    R: Array,
    mu0: Array,
    P0: Array,
    t_len: int,
    rng: np.random.Generator,
) -> Tuple[Array, Array]:
    state_dim = int(F.shape[0])
    x = np.zeros((t_len, state_dim), dtype=np.float64)
    x[0] = rng.multivariate_normal(np.asarray(mu0, dtype=np.float64), _psd(P0, eps=1e-12))
    for t in range(1, t_len):
        x[t] = F @ x[t - 1] + rng.multivariate_normal(
            np.zeros(state_dim), _psd(Q, eps=1e-12)
        )
    y = x @ np.asarray(H, dtype=np.float64).T
    y = y + rng.multivariate_normal(
        np.zeros(H.shape[0]), _psd(R, eps=1e-12), size=t_len
    )
    return x, y
