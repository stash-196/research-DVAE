"""Tresp & Hofmann nonlinear baseline for missing and noisy time series.

Formulation actually followed
----------------------------
The implemented prediction methods are those written in Tresp & Hofmann,
Neural Computation 10:731-747, 1998 (the journal version of the 1995 IEEE
Neural Networks for Signal Processing paper):

* Equation (1) / the appendix: ``y_t = f(y_{t-1}, ..., y_{t-N}) + eps`` with
  Gaussian ``eps``, or the first-order state form ``y_t = f(y_{t-1}) + eps``,
  ``z_t = g(y_t) + delta``. Missing entries are the infinite-measurement-noise
  case of section 3 (``sigma_delta^2 = inf``), implemented by skipping those
  coordinates in the update.
* Section 4.1 Gibbs sampling with sampling-importance-resampling from the
  univariate conditional given the Markov boundary. The proposal is the
  parent conditional ``N(f(parents), sigma^2)``; the importance weight is the
  product of successor likelihoods, which is the target/proposal ratio.
* Section 4.2 maximum-likelihood substitution: missing values minimize the
  sum of squared one-step residuals (their equation 9, joint over all missing
  values as in the Gaussian display that follows it).
* Section 4.3 extended Kalman filter: online first-order linearization of
  ``f``. Measurement variance of a present observation is a tiny jitter rather
  than exact zero, because ``R = 0`` makes the innovation covariance singular.
* Section 5.1 K-step prediction by forward simulation. The reported Monte
  Carlo predictor averages ``f`` over noisy rollouts. The iterated mean (noise
  off) is also returned, because that is the object an OTF free-run produces
  and because the paper plots both.

Training
--------
The 1998 paper says training under missing data "will be addressed in a
companion paper (Tresp and Hofmann, 1997)" and lists that companion as a
manuscript in preparation. The 1995 connectionists abstract says fundamental
equations for training are derived and approximated by stochastic simulation,
but that manuscript was not available here and the author's 2003 note says
they never followed it up. Those training equations are therefore not
implemented and are not invented here.

What the 1998 experiments actually do (sunspots, section 5.1.2) is train the
network on stretches where the inputs exist, with a weight-decay penalty, and
set ``sigma^2`` to the average squared residual. ``train_mode="observed_targets"``
is that protocol: least squares on timesteps whose delay window and target are
fully observed. The optimizer is not named in the paper; this code uses
Adam with ``weight_decay`` (classical L2 on the gradient, not AdamW). The
hidden-unit nonlinearity is also not named; the net is one hidden layer with
tanh, which is the usual MLP of the Bishop 1994 reference they cite. Hidden
size 0 is a linear map, used so tests can check an identifiable AR coefficient.
It is not a configuration from the paper.

``train_mode="multiple_imputation"`` is an extra loop, clearly not a
transcription of the unpublished companion. It alternates the observed-target
fit with the section 4.1 Gibbs fill. The paper explicitly identifies drawing
several imputations and averaging as multiple imputation (Little & Rubin),
which is why this mode exists, and it is off by default.

Multivariate series use the appendix's first-order model (delay 1): the state
is the observation vector. A delay stack on a multivariate series is not
written out in the paper and is rejected rather than guessed. Univariate
series use delay ``N`` as in equation (1).

A bootstrap particle filter is included under the name ``particle``. It is not
an algorithm in the 1998 paper. The author's 2003 comment on the 1995 version
says particle filters replaced the Gibbs sampler they proposed; the filter is
that later substitute and is not the primary reported trajectory.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from dvae.baselines.filtering.kalman_em import _psd, _solve_spd, _sym

Array = np.ndarray
PredictFn = Callable[[Array], Array]


def _as_2d(y: Array) -> Array:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError(f"Expected series shaped (T,) or (T, D), got {y.shape}")
    return y


def _observed_mask(y: Array, observed: Optional[Array]) -> Array:
    if observed is None:
        return np.isfinite(y)
    observed_b = np.asarray(observed, dtype=bool)
    if observed_b.ndim == 1:
        observed_b = observed_b.reshape(-1, 1)
    if observed_b.shape != y.shape:
        raise ValueError(
            f"observed shape {observed_b.shape} does not match y {y.shape}"
        )
    return observed_b & np.isfinite(y)


def _linear_fill(y: Array, observed: Array) -> Array:
    filled = y.copy()
    t_len, obs_dim = y.shape
    idx = np.arange(t_len)
    for d in range(obs_dim):
        good = observed[:, d]
        col = filled[:, d]
        if good.all():
            continue
        if not good.any():
            col[:] = 0.0
        else:
            col[~good] = np.interp(idx[~good], idx[good], col[good])
        filled[:, d] = col
    return filled


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        if hidden <= 0:
            self.net: nn.Module = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrespNonlinearModel:
    """Learned Gaussian transition with the 1998 prediction approximations."""

    def __init__(
        self,
        obs_dim: int,
        delay: int = 1,
        hidden: int = 16,
        r_obs: float = 1e-4,
        weight_decay: float = 0.0,
        seed: int = 0,
        standardize: bool = True,
    ) -> None:
        if obs_dim < 1:
            raise ValueError("obs_dim must be positive")
        if delay < 1:
            raise ValueError("delay must be >= 1")
        if obs_dim > 1 and delay != 1:
            raise ValueError(
                "Multivariate delay stacks are not specified in Tresp & Hofmann "
                "1998. Use delay=1 (appendix state-space form) when obs_dim > 1."
            )
        self.obs_dim = int(obs_dim)
        self.delay = int(delay)
        self.hidden = int(hidden)
        self.r_obs = float(r_obs)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.standardize = bool(standardize)
        self.in_dim = self.delay * self.obs_dim
        torch.manual_seed(self.seed)
        self.net = _MLP(self.in_dim, self.obs_dim, self.hidden).double()
        self.obs_mean = np.zeros(self.obs_dim, dtype=np.float64)
        self.obs_std = np.ones(self.obs_dim, dtype=np.float64)
        self.sigma2 = 1.0
        self.train_mode: Optional[str] = None

    def _scale(self, y: Array) -> Array:
        return (y - self.obs_mean) / self.obs_std

    def _unscale_obs(self, y_std: Array) -> Array:
        return y_std * self.obs_std.reshape(1, -1) + self.obs_mean.reshape(1, -1)

    def _fit_scale(self, y: Array, observed: Array) -> None:
        if not self.standardize:
            self.obs_mean = np.zeros(self.obs_dim)
            self.obs_std = np.ones(self.obs_dim)
            return
        mean = np.zeros(self.obs_dim)
        std = np.ones(self.obs_dim)
        for d in range(self.obs_dim):
            vals = y[observed[:, d], d]
            if vals.size:
                mean[d] = float(np.mean(vals))
                std[d] = float(np.std(vals))
        self.obs_mean = mean
        self.obs_std = np.where(std < 1e-8, 1.0, std)

    def _windows_targets(
        self, y: Array, observed: Array
    ) -> Tuple[Array, Array]:
        """Fully observed delay windows and targets, in standardized units."""
        xs: List[Array] = []
        ys: List[Array] = []
        t_len = y.shape[0]
        for t in range(self.delay, t_len):
            hist_ok = observed[t - self.delay : t].all()
            if not hist_ok or not observed[t].all():
                continue
            window = _window_at(y, t, self.delay)
            target = np.asarray(y[t], dtype=np.float64).reshape(-1)
            xs.append(np.asarray(window, dtype=np.float64).reshape(-1))
            ys.append(target)
        if not xs:
            raise ValueError(
                "No fully observed delay windows to train on. "
                "Use train_mode='multiple_imputation' or lower the delay."
            )
        return np.stack(xs), np.stack(ys)

    def _train_net(self, x: Array, target: Array, epochs: int, lr: float) -> None:
        xt = torch.tensor(x, dtype=torch.float64)
        yt = torch.tensor(target, dtype=torch.float64)
        opt = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=self.weight_decay
        )
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            pred = self.net(xt)
            loss = torch.mean((pred - yt) ** 2)
            loss.backward()
            opt.step()
        self.net.eval()
        with torch.no_grad():
            resid = self.net(xt) - yt
            self.sigma2 = float(torch.mean(resid**2).clamp_min(1e-8))

    def predict_mean_std(self, window_std: Array) -> Array:
        """One-step mean in standardized observation units."""
        w = np.asarray(window_std, dtype=np.float64).reshape(-1)
        if w.shape[0] != self.in_dim:
            raise ValueError(f"Expected window length {self.in_dim}, got {w.shape[0]}")
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(w, dtype=torch.float64))
        return out.numpy().reshape(self.obs_dim)

    def jacobian_std(self, window_std: Array) -> Array:
        """Jacobian of the standardized one-step mean, shape (obs_dim, in_dim)."""
        w = torch.tensor(
            np.asarray(window_std, dtype=np.float64).reshape(-1), dtype=torch.float64
        )

        def _f(v: torch.Tensor) -> torch.Tensor:
            return self.net(v)

        jac = torch.autograd.functional.jacobian(_f, w)
        return jac.detach().numpy().reshape(self.obs_dim, self.in_dim)

    def fit(
        self,
        y: Array,
        observed: Optional[Array] = None,
        *,
        train_mode: str = "observed_targets",
        epochs: int = 50,
        lr: float = 1e-2,
        n_outer: int = 3,
        gibbs_sweeps: int = 4,
        gibbs_burn: int = 1,
        n_sir: int = 16,
    ) -> "TrespNonlinearModel":
        if train_mode not in ("observed_targets", "multiple_imputation"):
            raise ValueError(
                "train_mode must be 'observed_targets' or 'multiple_imputation'"
            )
        y = _as_2d(y)
        if y.shape[1] != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {y.shape[1]}")
        observed_b = _observed_mask(y, observed)
        self._fit_scale(y, observed_b)
        y_std = self._scale(y)
        self.train_mode = train_mode
        if train_mode == "observed_targets":
            inputs, targets = self._windows_targets(y_std, observed_b)
            self._train_net(inputs, targets, epochs=epochs, lr=lr)
            return self

        filled = self._scale(_linear_fill(y, observed_b))
        # Interpolated values are a complete stand-in so every window trains.
        full_obs = np.ones_like(observed_b)
        rng = np.random.default_rng(self.seed)
        for outer in range(n_outer):
            inputs, targets = self._windows_targets(filled, full_obs)
            inner_epochs = epochs if outer == 0 else max(epochs // 2, 1)
            self._train_net(inputs, targets, epochs=inner_epochs, lr=lr)
            samples = gibbs_impute(
                self.predict_mean_std,
                filled,
                observed_b,
                delay=self.delay,
                sigma=float(np.sqrt(self.sigma2)),
                n_sweeps=gibbs_sweeps,
                burn=gibbs_burn,
                n_sir=n_sir,
                rng=rng,
            )
            filled = samples.mean(axis=0)
            # Keep genuine observations clamped.
            filled = np.where(observed_b, y_std, filled)
        inputs, targets = self._windows_targets(filled, full_obs)
        self._train_net(inputs, targets, epochs=max(epochs // 2, 1), lr=lr)
        return self

    def linear_weight(self) -> Array:
        """Weight matrix of a hidden=0 map. Raises if the net is not linear."""
        layer = self.net.net
        if not isinstance(layer, nn.Linear):
            raise ValueError("linear_weight is only defined for hidden=0")
        return layer.weight.detach().numpy().copy()

    def impute(
        self,
        y: Array,
        observed: Optional[Array] = None,
        *,
        auto_mask: Optional[Array] = None,
        n_particles: int = 0,
        n_samples: int = 32,
        gibbs_sweeps: int = 6,
        gibbs_burn: int = 2,
        n_sir: int = 24,
        do_ml: bool = True,
        do_gibbs: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Array]:
        """Causal EKF trajectories plus the paper's acausal imputations.

        ``predicted`` / ``filtered`` come from the EKF (section 4.3).
        ``autonomous`` is the noiseless iteration after ``auto_mask`` (the OTF
        free-run analogue; the paper's "iterated system").
        ``autonomous_sampled`` averages ``f`` over noisy K-step rollouts
        (section 5.1) and is not the OTF comparison trajectory.
        ``gibbs`` is the posterior mean from section 4.1 (uses the future).
        ``ml`` is the section 4.2 substitution.
        """
        y = _as_2d(y)
        observed_b = _observed_mask(y, observed)
        if y.shape[1] != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {y.shape[1]}")
        rng = np.random.default_rng(self.seed) if rng is None else rng
        y_std = self._scale(y)
        sigma = float(np.sqrt(max(self.sigma2, 1e-8)))
        # r_obs is a fraction of standardized variance. The paper uses 0 for a
        # perfect measurement; exact 0 is singular, so this is a jitter.
        r_std = max(self.r_obs, 1e-8)
        causal = ekf_trajectories(
            predict_mean=self.predict_mean_std,
            jacobian=self.jacobian_std,
            y=y_std,
            observed=observed_b,
            delay=self.delay,
            sigma2=self.sigma2,
            r_obs=r_std,
        )
        out: Dict[str, Array] = {
            "predicted": self._unscale_obs(causal["predicted"]),
            "filtered": self._unscale_obs(causal["filtered"]),
            "train_mode": self.train_mode,
            "sigma2_standardized": self.sigma2,
        }
        if auto_mask is not None:
            auto_mask = np.asarray(auto_mask, dtype=bool).reshape(-1)
            blocked = observed_b & ~auto_mask.reshape(-1, 1)
            rolled = ekf_trajectories(
                predict_mean=self.predict_mean_std,
                jacobian=self.jacobian_std,
                y=y_std,
                observed=blocked,
                delay=self.delay,
                sigma2=self.sigma2,
                r_obs=r_std,
            )
            out["autonomous"] = self._unscale_obs(rolled["predicted"])
            flip = int(np.flatnonzero(auto_mask)[0]) if auto_mask.any() else y.shape[0]
            hist = causal["filtered"][max(0, flip - self.delay) : flip]
            if hist.shape[0] < self.delay:
                pad = np.zeros((self.delay - hist.shape[0], self.obs_dim))
                hist = np.concatenate([pad, hist], axis=0)
            n_steps = y.shape[0] - flip
            if n_steps > 0:
                iterated, sampled = k_step_predictions(
                    self.predict_mean_std,
                    hist,
                    n_steps=n_steps,
                    n_samples=n_samples,
                    sigma=sigma,
                    rng=rng,
                )
                # The EKF open-loop above is the filter's iterated mean and is
                # what ``autonomous`` stores. Section 5.1's Monte Carlo mean
                # is reported separately, in original units.
                out["autonomous_sampled"] = self._unscale_obs(sampled)
                out["autonomous_iterated_from_boundary"] = self._unscale_obs(iterated)
        if do_gibbs:
            samples = gibbs_impute(
                self.predict_mean_std,
                y_std,
                observed_b,
                delay=self.delay,
                sigma=sigma,
                n_sweeps=gibbs_sweeps,
                burn=gibbs_burn,
                n_sir=n_sir,
                rng=rng,
            )
            gibbs_mean = np.where(observed_b, y_std, samples.mean(axis=0))
            out["gibbs"] = self._unscale_obs(gibbs_mean)
        if do_ml:
            ml = ml_substitute(
                self.predict_mean_std,
                y_std,
                observed_b,
                delay=self.delay,
            )
            out["ml"] = self._unscale_obs(ml)
        if n_particles > 0:
            pf = particle_filter(
                self.predict_mean_std,
                y_std,
                observed_b,
                delay=self.delay,
                sigma=sigma,
                r_obs=r_std,
                n_particles=n_particles,
                rng=rng,
            )
            out["particle"] = self._unscale_obs(pf)
        return out


def _window_at(series: Array, t: int, delay: int) -> Array:
    """Most-recent-first window of observations that predict time ``t``."""
    # series[t-1], series[t-2], ... 
    parts = [series[t - 1 - i] for i in range(delay)]
    return np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts])


def gibbs_impute(
    predict_mean: PredictFn,
    y: Array,
    observed: Array,
    *,
    delay: int,
    sigma: float,
    n_sweeps: int,
    burn: int,
    n_sir: int,
    rng: np.random.Generator,
) -> Array:
    """Section 4.1. Returns samples shaped (n_keep, T, D).

    The chain is initialized by linear interpolation. Times without a full
    parent window (the first ``delay`` steps) are not given a model conditional
    in equation (1); they stay at the interpolated fill. The paper leaves
    ``P(y_1, ..., y_N)`` unspecified.
    """
    y = _as_2d(y)
    observed = _observed_mask(y, observed)
    state = _linear_fill(y, observed)
    t_len, obs_dim = y.shape
    missing = np.argwhere(~observed)
    # Only coordinates that participate in at least one full window.
    missing = [tuple(ix) for ix in missing if ix[0] >= delay]
    kept: List[Array] = []
    for sweep in range(burn + n_sweeps):
        for t, d in missing:
            state[t, d] = _sir_one(
                predict_mean, state, t, d, delay, sigma, n_sir, rng
            )
        if sweep >= burn:
            kept.append(state.copy())
    if not kept:
        kept.append(state.copy())
    return np.stack(kept, axis=0)


def _sir_one(
    predict_mean: PredictFn,
    state: Array,
    t: int,
    d: int,
    delay: int,
    sigma: float,
    n_sir: int,
    rng: np.random.Generator,
) -> float:
    window = _window_at(state, t, delay)
    parent = predict_mean(window)
    proposals = rng.normal(parent[d], sigma, size=n_sir)
    log_w = np.zeros(n_sir, dtype=np.float64)
    t_len = state.shape[0]
    var = sigma**2
    for j in range(1, delay + 1):
        child_t = t + j
        if child_t >= t_len or child_t < delay:
            continue
        base_window = _window_at(state, child_t, delay)
        # y_t sits at lag j, which is index (j * obs_dim + d) in the
        # most-recent-first window? lag 1 is the first obs vector.
        # window = [y_{child-1}, y_{child-2}, ...]
        # y_t = y_{child-j} is observation-vector slot (j - 1).
        obs_dim = state.shape[1]
        slot = (j - 1) * obs_dim + d
        for s, cand in enumerate(proposals):
            w = base_window.copy()
            w[slot] = cand
            mean = predict_mean(w)
            resid = state[child_t] - mean
            # Parent residual is in the proposal, so the weight is the child
            # likelihood only (section 4.1 SIR).
            log_w[s] += -0.5 * float(np.dot(resid, resid)) / var
    log_w -= log_w.max()
    weights = np.exp(log_w)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return float(proposals[0])
    weights /= total
    choice = int(rng.choice(n_sir, p=weights))
    return float(proposals[choice])


def ml_substitute(
    predict_mean: PredictFn,
    y: Array,
    observed: Array,
    *,
    delay: int,
    max_free: int = 400,
) -> Array:
    """Section 4.2 joint least-squares substitution of the missing entries.

    Series longer than ``max_free`` missing scalars are optimized in successive
    chunks. The paper states a joint optimization and does not discuss
    chunking; chunking is only a limit on the optimizer.
    """
    from scipy.optimize import minimize

    y = _as_2d(y)
    observed = _observed_mask(y, observed)
    filled = _linear_fill(y, observed)
    missing_ix = np.argwhere(~observed)
    if missing_ix.size == 0:
        return filled
    # Drop leading indices that never enter f as a target or a parent of a
    # later target inside the series. They still appear if a later window
    # sees them; keep all missing indices at or after time 0 that are used.
    def objective(free: Array, indices: Array, current: Array) -> float:
        trial = current.copy()
        for val, (t, d) in zip(free, indices):
            trial[t, d] = val
        err = 0.0
        for t in range(delay, trial.shape[0]):
            mean = predict_mean(_window_at(trial, t, delay))
            resid = trial[t] - mean
            err += float(np.dot(resid, resid))
        return err

    cursor = 0
    while cursor < len(missing_ix):
        chunk = missing_ix[cursor : cursor + max_free]
        x0 = np.array([filled[t, d] for t, d in chunk], dtype=np.float64)
        result = minimize(
            objective,
            x0,
            args=(chunk, filled),
            method="L-BFGS-B",
            options={"maxiter": 50},
        )
        for val, (t, d) in zip(result.x, chunk):
            filled[t, d] = val
        cursor += max_free
    # Clamp observations. The objective does not move them, but be explicit.
    filled = np.where(observed, y, filled)
    return filled


def k_step_predictions(
    predict_mean: PredictFn,
    history: Array,
    *,
    n_steps: int,
    n_samples: int,
    sigma: float,
    rng: np.random.Generator,
) -> Tuple[Array, Array]:
    """Section 5.1. ``history`` is chronological, shape (delay, obs_dim).

    Returns ``(iterated_mean, monte_carlo_mean)``, each (n_steps, obs_dim).
    The Monte Carlo predictor averages ``f`` over noisy forward samples. The
    iterated mean is one noiseless rollout. The paper shows these differ once
    ``f`` is nonlinear and ``K > 1``.
    """
    history = np.asarray(history, dtype=np.float64)
    if history.ndim == 1:
        history = history.reshape(-1, 1)
    delay, obs_dim = history.shape
    iterated = np.zeros((n_steps, obs_dim), dtype=np.float64)
    hist = [history[i].copy() for i in range(delay)]
    for k in range(n_steps):
        mean = predict_mean(_window_from_list(hist, delay))
        iterated[k] = mean
        hist.append(mean.copy())
        hist = hist[-delay:]
    acc = np.zeros((n_steps, obs_dim), dtype=np.float64)
    for _s in range(n_samples):
        hist = [history[i].copy() for i in range(delay)]
        for k in range(n_steps):
            mean = predict_mean(_window_from_list(hist, delay))
            acc[k] += mean
            hist.append(mean + rng.normal(0.0, sigma, size=obs_dim))
            hist = hist[-delay:]
    return iterated, acc / max(n_samples, 1)


def _window_from_list(hist: Sequence[Array], delay: int) -> Array:
    parts = [hist[-1 - i] for i in range(delay)]
    return np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts])


def ekf_trajectories(
    predict_mean: PredictFn,
    jacobian: Callable[[Array], Array],
    y: Array,
    observed: Array,
    *,
    delay: int,
    sigma2: float,
    r_obs: float,
) -> Dict[str, Array]:
    """Section 4.3 EKF on the delay state.

    State at time t is ``[y_t, y_{t-1}, ..., y_{t-delay+1}]`` flattened in
    observation-vector blocks, most recent first. Process noise sits on the
    newly predicted block. Lag blocks are copies; a 1e-8 jitter keeps the
    covariance positive definite (the paper's noise is only on ``y_t``).
    """
    y = _as_2d(y)
    t_len, obs_dim = y.shape
    state_dim = delay * obs_dim
    q = np.eye(state_dim, dtype=np.float64) * 1e-8
    q[:obs_dim, :obs_dim] = np.eye(obs_dim) * max(sigma2, 1e-8)
    eye = np.eye(state_dim, dtype=np.float64)
    x_filt = np.zeros(state_dim, dtype=np.float64)
    p_filt = np.eye(state_dim, dtype=np.float64)
    predicted = np.zeros((t_len, obs_dim), dtype=np.float64)
    filtered = np.zeros((t_len, obs_dim), dtype=np.float64)

    def _transit(state: Array) -> Array:
        mean_y = predict_mean(state)
        if delay == 1:
            return mean_y.reshape(-1)
        nxt = np.empty(state_dim, dtype=np.float64)
        nxt[:obs_dim] = mean_y
        nxt[obs_dim:] = state[:-obs_dim]
        return nxt

    def _jac(state: Array) -> Array:
        dy = jacobian(state)
        if delay == 1:
            return dy
        jac = np.zeros((state_dim, state_dim), dtype=np.float64)
        jac[:obs_dim, :] = dy
        jac[obs_dim:, :-obs_dim] = np.eye(state_dim - obs_dim)
        return jac

    for t in range(t_len):
        if t == 0:
            xp = x_filt.copy()
            pp = p_filt.copy()
        else:
            jac = _jac(x_filt)
            xp = _transit(x_filt)
            pp = _psd(jac @ p_filt @ jac.T + q, eps=1e-10)
        predicted[t] = xp[:obs_dim]
        idx = np.flatnonzero(observed[t])
        if idx.size == 0:
            x_filt = xp
            p_filt = pp
        else:
            h = np.zeros((idx.size, state_dim), dtype=np.float64)
            for row, comp in enumerate(idx):
                h[row, comp] = 1.0
            rt = np.eye(idx.size) * r_obs
            innov_cov = _sym(h @ pp @ h.T + rt)
            gain = _solve_spd(innov_cov, h @ pp).T
            innov = y[t, idx] - h @ xp
            x_filt = xp + gain @ innov
            ikh = eye - gain @ h
            p_filt = _psd(ikh @ pp @ ikh.T + gain @ rt @ gain.T, eps=1e-12)
        filtered[t] = x_filt[:obs_dim]
    return {"predicted": predicted, "filtered": filtered}


def particle_filter(
    predict_mean: PredictFn,
    y: Array,
    observed: Array,
    *,
    delay: int,
    sigma: float,
    r_obs: float,
    n_particles: int,
    rng: np.random.Generator,
) -> Array:
    """Bootstrap particle filter. Not in the 1998 equations; see the module note."""
    y = _as_2d(y)
    t_len, obs_dim = y.shape
    state_dim = delay * obs_dim
    particles = rng.normal(0.0, 1.0, size=(n_particles, state_dim))
    weights = np.full(n_particles, 1.0 / n_particles)
    means = np.zeros((t_len, obs_dim), dtype=np.float64)

    def _transit_one(state: Array, noise: Array) -> Array:
        mean_y = predict_mean(state) + noise
        if delay == 1:
            return mean_y
        nxt = np.empty(state_dim, dtype=np.float64)
        nxt[:obs_dim] = mean_y
        nxt[obs_dim:] = state[:-obs_dim]
        return nxt

    for t in range(t_len):
        if t > 0:
            noise = rng.normal(0.0, sigma, size=(n_particles, obs_dim))
            particles = np.stack(
                [_transit_one(particles[i], noise[i]) for i in range(n_particles)],
                axis=0,
            )
        idx = np.flatnonzero(observed[t])
        if idx.size:
            resid = particles[:, idx] - y[t, idx]
            log_lik = -0.5 * np.sum(resid**2, axis=1) / max(r_obs, 1e-8)
            log_lik -= log_lik.max()
            weights = weights * np.exp(log_lik)
            total = weights.sum()
            if not np.isfinite(total) or total <= 0:
                weights[:] = 1.0 / n_particles
            else:
                weights /= total
            ess = 1.0 / np.sum(weights**2)
            if ess < n_particles / 2.0:
                choice = rng.choice(n_particles, size=n_particles, p=weights)
                particles = particles[choice]
                weights[:] = 1.0 / n_particles
        means[t] = weights @ particles[:, :obs_dim]
    return means
