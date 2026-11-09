#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SINDy baseline (Brunton, Proctor, Kutz, PNAS 2016).

Fits a sparse polynomial vector field with sequential thresholded least
squares. The default backend is a small NumPy STLSQ so training containers
do not need pysindy. ``backend="pysindy"`` uses the pysindy package when it
is installed (``pip install 'research_dvae[sindy]'``).

Missing measurements
--------------------
``observed_segments`` (default): a row is kept only when every coordinate is
finite. Contiguous runs are differentiated separately and pooled into one
regression, so the derivative is never taken across a gap.

``interpolate``: each coordinate is linearly interpolated, then SINDy is fit
on the filled series. That fills holes with invented samples and is
optimistic when gaps are long.

Partial observation (for example Lorenz ``only_x``) is a different limit.
The fit is in the coordinates that exist. A 1-D or 2-D polynomial ODE does
not contain the ``xy`` / ``xz`` coupling of Lorenz63, so the identified
field cannot reproduce the attractor. Fully observed 3-D series are the
regime where SINDy is expected to do well.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def polynomial_library(states, degree):
    """Polynomial features up to ``degree``, including the constant term.

    Columns are ``1``, then the degree-1 coordinates ``x0, x1, ...``, then
    higher monomials in nondecreasing index order (``x0*x0``, ``x0*x1``, ...).
    ``states`` is ``(n, d)`` or ``(d,)``.
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states.reshape(-1, 1)
    n_rows, n_state = states.shape
    if degree < 1:
        raise ValueError("degree must be >= 1")
    columns = [np.ones(n_rows)]
    names = ["1"]

    def _monomials(start, remaining, chosen):
        if remaining == 0:
            column = np.ones(n_rows)
            label = []
            for idx in chosen:
                column = column * states[:, idx]
                label.append(f"x{idx}")
            columns.append(column)
            names.append("*".join(label))
            return
        for j in range(start, n_state):
            _monomials(j, remaining - 1, chosen + [j])

    for deg in range(1, int(degree) + 1):
        _monomials(0, deg, [])
    return np.column_stack(columns), names


def finite_difference(states, dt):
    """Central differences along axis 0. Same length as ``states``."""
    return np.gradient(np.asarray(states, dtype=np.float64), float(dt), axis=0)


def stlsq(theta, targets, threshold, max_iter=10):
    """Sequential thresholded least squares (Brunton et al.).

    ``theta`` is ``(n, n_features)``, ``targets`` is ``(n, n_states)``.
    Returns coefficients ``(n_features, n_states)``.
    """
    theta = np.asarray(theta, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[:, None]
    coef = np.linalg.lstsq(theta, targets, rcond=None)[0]
    threshold = float(threshold)
    for _ in range(int(max_iter)):
        small = np.abs(coef) < threshold
        updated = np.zeros_like(coef)
        for k in range(targets.shape[1]):
            keep = ~small[:, k]
            if not np.any(keep):
                continue
            updated[keep, k] = np.linalg.lstsq(theta[:, keep], targets[:, k], rcond=None)[0]
        if np.allclose(updated, coef, rtol=0.0, atol=0.0):
            coef = updated
            break
        coef = updated
    return coef


def contiguous_observed_segments(states, min_len):
    """Split ``(T, D)`` into runs where every coordinate is finite."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[:, None]
    observed = np.isfinite(states).all(axis=1)
    segments = []
    start = 0
    n_rows = len(states)
    while start < n_rows:
        if not observed[start]:
            start += 1
            continue
        stop = start + 1
        while stop < n_rows and observed[stop]:
            stop += 1
        if stop - start >= min_len:
            segments.append(states[start:stop])
        start = stop
    n_dropped = int((~observed).sum())
    return segments, n_dropped


def interpolate_missing(states):
    """Linear interpolation along time, independently per coordinate.

    Ends use the nearest finite sample (``np.interp`` endpoint rule). A
    coordinate that is entirely missing becomes zero.
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[:, None]
    filled = np.array(states, copy=True)
    time = np.arange(len(filled))
    for d in range(filled.shape[1]):
        column = filled[:, d]
        valid = np.isfinite(column)
        if not np.any(valid):
            filled[:, d] = 0.0
        elif valid.sum() == 1:
            filled[:, d] = column[valid][0]
        else:
            filled[:, d] = np.interp(time, time[valid], column[valid])
    return filled


def drop_unobserved_columns(states):
    """Remove coordinates that are NaN at every time. Returns ``(kept, index)``."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[:, None]
    keep = np.isfinite(states).any(axis=0)
    if not np.any(keep):
        raise ValueError("SINDy input has no finite samples.")
    index = np.flatnonzero(keep)
    return states[:, keep], index


def _limits_text(state_dim, missing_strategy, n_dropped, n_segments):
    parts = []
    if state_dim < 3:
        parts.append(
            "SINDy is identified in the observed coordinates only "
            f"({state_dim}-D). Lorenz63's attractor needs the x, y, z coupling "
            "(the xy and xz terms). A polynomial ODE on x alone, or on a pair "
            "of coordinates, cannot represent that vector field. Expect a "
            "faithful sparse model on fully observed 3-D series and a clear "
            "drop under only_x / only_xy / only_xz. Delay embedding is not "
            "applied; that would be a different model."
        )
    else:
        parts.append(
            "State dimension is 3, so the polynomial library can represent "
            "the Lorenz63 vector field. On clean, fully observed data, "
            "degree-2 STLSQ is expected to recover sigma, rho, and beta."
        )
    if missing_strategy == "observed_segments":
        parts.append(
            "Missing values use observed segments: any row with a NaN is "
            "dropped, contiguous runs of length at least the minimum are "
            f"differentiated on their own, and those runs are pooled "
            f"({n_segments} segments, {n_dropped} rows dropped). The "
            "derivative is not taken across a gap. Long gaps shrink the "
            "sample and can leave no usable segment."
        )
    elif missing_strategy == "interpolate":
        parts.append(
            "Missing values are linearly interpolated per coordinate before "
            "the fit. Interpolated samples are invented and the derivative "
            "crosses the gap, so this is optimistic when outages are long. "
            f"Rows that were incomplete before filling: {n_dropped}."
        )
    else:
        raise ValueError(f"unknown missing strategy {missing_strategy}")
    parts.append(
        "Autonomous scores use the same names as the OTF eval "
        "(mse_auto, kld_auto, spectrum_error_auto): integrate from the last "
        "fully observed state at the half-series flip, then compare the free "
        "run to the held-out tail. On Lorenz, full-horizon MSE grows even "
        "for a correct field because of chaos; the spectrum Hellinger "
        "distance, the delay-embedding KL, and the short-horizon MSE are the "
        "scores that stay informative."
    )
    return " ".join(parts)


@dataclass
class SindyFit:
    """Sparse polynomial vector field ``dx/dt = Theta(x) Xi``."""

    coefficients: np.ndarray
    feature_names: list
    state_dim: int
    degree: int
    threshold: float
    backend: str
    missing_strategy: str
    n_segments: int
    n_rows: int
    n_dropped: int
    partial_observation: bool
    column_index: np.ndarray
    limits: str
    dt: float
    _pysindy_model: object = field(default=None, repr=False)

    def equations(self):
        lines = []
        for k in range(self.state_dim):
            terms = []
            for name, value in zip(self.feature_names, self.coefficients[:, k]):
                if value == 0.0:
                    continue
                terms.append(f"{value:+.6g} {name}")
            rhs = " ".join(terms) if terms else "0"
            lines.append(f"dx{k}/dt = {rhs}")
        return "\n".join(lines)

    def predict_derivative(self, states):
        states = np.asarray(states, dtype=np.float64)
        single = states.ndim == 1
        if single:
            states = states.reshape(1, -1)
        if self._pysindy_model is not None:
            deriv = np.asarray(self._pysindy_model.predict(states), dtype=np.float64)
        else:
            theta, _ = polynomial_library(states, self.degree)
            deriv = theta @ self.coefficients
        if single:
            return deriv.reshape(-1)
        return deriv

    def simulate(self, x0, n_steps, dt=None):
        """Integrate ``n_steps`` samples including the initial condition.

        Divergent trajectories are filled with NaN from the failing step.
        """
        dt = self.dt if dt is None else float(dt)
        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0.shape != (self.state_dim,):
            raise ValueError(
                f"initial condition width {x0.shape} != state_dim {self.state_dim}"
            )
        if self._pysindy_model is not None:
            time = np.arange(n_steps) * dt
            try:
                traj = np.asarray(self._pysindy_model.simulate(x0, time), dtype=np.float64)
            except Exception:
                traj = np.full((n_steps, self.state_dim), np.nan)
            if traj.shape[0] != n_steps:
                padded = np.full((n_steps, self.state_dim), np.nan)
                take = min(n_steps, traj.shape[0])
                padded[:take] = traj[:take]
                traj = padded
            return traj
        return _rk4(self, x0, n_steps, dt)


def _rk4(fit, x0, n_steps, dt):
    traj = np.empty((n_steps, fit.state_dim), dtype=np.float64)
    traj[0] = x0
    for i in range(n_steps - 1):
        state = traj[i]
        k1 = fit.predict_derivative(state).reshape(-1)
        k2 = fit.predict_derivative(state + 0.5 * dt * k1).reshape(-1)
        k3 = fit.predict_derivative(state + 0.5 * dt * k2).reshape(-1)
        k4 = fit.predict_derivative(state + dt * k3).reshape(-1)
        nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if (not np.isfinite(nxt).all()) or np.max(np.abs(nxt)) > 1.0e4:
            traj[i + 1 :] = np.nan
            break
        traj[i + 1] = nxt
    return traj


def _prepare_regression(segments, dt, degree):
    thetas = []
    targets = []
    names = None
    for segment in segments:
        if len(segment) < 3:
            continue
        deriv = finite_difference(segment, dt)
        theta, names = polynomial_library(segment, degree)
        # End samples of a central difference are one-sided; drop them.
        thetas.append(theta[1:-1])
        targets.append(deriv[1:-1])
    if not thetas:
        raise ValueError("no segment long enough to differentiate")
    return np.concatenate(thetas, axis=0), np.concatenate(targets, axis=0), names


def _fit_pysindy(segments, dt, degree, threshold):
    try:
        import pysindy as ps
    except ImportError as exc:
        raise ImportError(
            "pysindy is optional and is not installed. "
            "Use backend='numpy' (the default STLSQ) or "
            "pip install 'research_dvae[sindy]'."
        ) from exc
    model = ps.SINDy(
        differentiation_method=ps.FiniteDifference(),
        feature_library=ps.PolynomialLibrary(degree=degree, include_bias=True),
        optimizer=ps.STLSQ(threshold=threshold),
    )
    if len(segments) == 1:
        model.fit(segments[0], t=float(dt))
    else:
        model.fit(segments, t=float(dt), multiple_trajectories=True)
    return model


def fit_sindy(
    trajectories,
    dt,
    *,
    degree=2,
    threshold=0.1,
    missing_strategy="observed_segments",
    backend="numpy",
    min_segment_len=8,
):
    """Fit SINDy on one ``(T, D)`` array or a list of them.

    ``missing_strategy`` is ``observed_segments`` or ``interpolate``.
    ``backend`` is ``numpy`` or ``pysindy``.
    """
    if isinstance(trajectories, np.ndarray):
        trajectories = [trajectories]
    pieces = [np.asarray(piece, dtype=np.float64) for piece in trajectories]
    if not pieces:
        raise ValueError("no trajectories")
    column_index = None
    prepared = []
    n_dropped = 0
    for piece in pieces:
        if piece.ndim == 1:
            piece = piece[:, None]
        kept, index = drop_unobserved_columns(piece)
        if column_index is None:
            column_index = index
        elif not np.array_equal(index, column_index):
            raise ValueError("trajectories disagree on which coordinates are observed")
        n_incomplete = int((~np.isfinite(kept).all(axis=1)).sum())
        n_dropped += n_incomplete
        if missing_strategy == "interpolate":
            prepared.append(interpolate_missing(kept))
        elif missing_strategy == "observed_segments":
            prepared.append(kept)
        else:
            raise ValueError(
                "missing_strategy must be 'observed_segments' or 'interpolate'"
            )
    if missing_strategy == "interpolate":
        segments = [piece for piece in prepared if len(piece) >= min_segment_len]
        # interpolate already filled holes; still count them via n_dropped.
    else:
        segments = []
        dropped_rows = 0
        for piece in prepared:
            runs, n_drop = contiguous_observed_segments(piece, min_segment_len)
            segments.extend(runs)
            dropped_rows += n_drop
        n_dropped = dropped_rows
    if not segments:
        raise ValueError(
            "SINDy found no contiguous observed segment. "
            "Lower min_segment_len, or use missing_strategy='interpolate'."
        )
    state_dim = int(segments[0].shape[1])
    n_rows = int(sum(len(seg) for seg in segments))
    pysindy_model = None
    if backend == "numpy":
        theta, target, names = _prepare_regression(segments, dt, degree)
        if theta.shape[0] < theta.shape[1]:
            raise ValueError(
                f"not enough finite samples ({theta.shape[0]}) for a degree-{degree} "
                f"library with {theta.shape[1]} columns"
            )
        coefficients = stlsq(theta, target, threshold)
    elif backend == "pysindy":
        pysindy_model = _fit_pysindy(segments, dt, degree, threshold)
        names = [str(name) for name in pysindy_model.get_feature_names()]
        raw = np.atleast_2d(np.asarray(pysindy_model.coefficients(), dtype=np.float64))
        if raw.shape == (len(names), state_dim):
            coefficients = raw
        elif raw.shape == (state_dim, len(names)):
            coefficients = raw.T
        else:
            raise RuntimeError(
                f"unexpected pysindy coefficient shape {raw.shape} "
                f"for {len(names)} features and state_dim {state_dim}"
            )
    else:
        raise ValueError("backend must be 'numpy' or 'pysindy'")
    partial = state_dim < 3
    return SindyFit(
        coefficients=coefficients,
        feature_names=list(names),
        state_dim=state_dim,
        degree=int(degree),
        threshold=float(threshold),
        backend=backend,
        missing_strategy=missing_strategy,
        n_segments=len(segments),
        n_rows=n_rows,
        n_dropped=int(n_dropped),
        partial_observation=partial,
        column_index=np.asarray(column_index),
        limits=_limits_text(state_dim, missing_strategy, int(n_dropped), len(segments)),
        dt=float(dt),
        _pysindy_model=pysindy_model,
    )


def autonomous_segment(window, fit, flip_point=None):
    """Free-run a window from the last fully observed state at the flip.

    Returns ``(gt_auto, pred_auto, flip_point)``. ``gt_auto`` may still
    contain NaNs. Returns ``None`` when the prefix has no complete state
    or the forecast is empty.
    """
    window = np.asarray(window, dtype=np.float64)
    if window.ndim == 1:
        window = window[:, None]
    if window.shape[1] != fit.state_dim:
        # Allow a wider Lorenz array only when the fit stored a column index
        # into it. Callers should already have sliced.
        raise ValueError(
            f"window width {window.shape[1]} != SINDy state_dim {fit.state_dim}"
        )
    n_rows = len(window)
    if flip_point is None:
        flip_point = n_rows // 2
    flip_point = int(flip_point)
    if flip_point <= 0 or flip_point >= n_rows:
        return None
    ic_index = None
    for t in range(flip_point - 1, -1, -1):
        if np.isfinite(window[t]).all():
            ic_index = t
            break
    if ic_index is None:
        return None
    n_steps = n_rows - ic_index
    traj = fit.simulate(window[ic_index], n_steps, dt=fit.dt)
    # traj[0] is the initial condition at ic_index. The OTF half-series
    # score starts at flip_point, not at the (possibly earlier) IC.
    pred = traj[(flip_point - ic_index) :]
    gt = window[flip_point:]
    if len(pred) == 0 or len(gt) == 0:
        return None
    return gt, pred[: len(gt)], flip_point


def _mean_key(rows, key):
    values = []
    for row in rows:
        value = row.get(key, np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan")
    return float(np.mean(values))


def score_rollouts(fit, windows, dataset_name="Lorenz63"):
    """Score half-series free runs with the OTF autonomous metric names.

    ``windows`` are ``(T, state_dim)`` arrays, possibly with NaNs. The
    returned dict uses ``mse_auto``, ``kld_auto``, and ``spectrum_error_auto``
    (means over windows and channels) so a summary can sit next to an OTF
    ``evaluation_summary.yaml``.
    """
    from dvae.baselines.rollout_metrics import score_trajectory_pair

    rows = []
    for window in windows:
        piece = autonomous_segment(window, fit)
        if piece is None:
            continue
        gt, pred, flip_point = piece
        metrics = score_trajectory_pair(
            gt,
            pred,
            dataset_name=dataset_name,
            dt=fit.dt,
            prefix="auto",
        )
        metrics["flip_point"] = int(flip_point)
        metrics["diverged"] = bool(not np.isfinite(pred).all())
        metrics["gt_auto"] = gt
        metrics["pred_auto"] = pred
        rows.append(metrics)
    summary = {
        "n_windows": len(rows),
        "n_diverged": int(sum(1 for row in rows if row["diverged"])),
        "mse_auto": _mean_key(rows, "mse_auto"),
        "mse_auto_mean": _mean_key(rows, "mse_auto_mean"),
        "mse_auto_short": _mean_key(rows, "mse_auto_short"),
        "kld_auto": _mean_key(rows, "kld_auto"),
        "kld_auto_mean": _mean_key(rows, "kld_auto_mean"),
        "spectrum_error_auto": _mean_key(rows, "spectrum_error_auto"),
        "spectrum_error_auto_mean": _mean_key(rows, "spectrum_error_auto_mean"),
        "windows": rows,
    }
    return summary
