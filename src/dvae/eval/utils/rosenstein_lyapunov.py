"""Rosenstein nearest-neighbour estimate of the largest Lyapunov exponent.

Rosenstein, Collins, and De Luca (1993): delay-embed a scalar series, pair
each point with its nearest neighbour outside a Theiler window, and take the
slope of the mean log divergence. The slope is in 1/time when ``dt`` is the
sample interval.

No Numba. Neighbour search is a scipy KD-tree.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
from scipy.spatial import cKDTree

from dvae.eval.utils.intrinsic_dim import delay_dim_for_key

# Follow pairs long enough to leave the initial stretching transient.
# On a dt=0.01 Lorenz series the scaling region sits near 0.5–2 time units.
DEFAULT_EVOLUTION = 150
DEFAULT_MIN_PAIRS = 30
# Cap pair count so a long free-run stays cheap. Fixed seed keeps the slope stable.
DEFAULT_MAX_PAIRS = 5000
_PAIR_SAMPLE_SEED = 0
# Distances below this are treated as ties / numerical zeros.
_DIST_FLOOR = 1e-12


def longest_finite_segment(series: np.ndarray) -> np.ndarray:
    """Longest contiguous run of finite samples. NaNs break time, so they are not dropped in place."""
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    if not finite.any():
        return x[:0]
    padded = np.concatenate([[False], finite, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return x[int(starts[best]) : int(ends[best])]


def delay_embed(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Time-delay embedding. ``series`` must already be finite and 1D."""
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    m = int(m)
    tau = int(tau)
    if m < 1 or tau < 1:
        raise ValueError(f"embed dim and tau must be >= 1, got m={m}, tau={tau}")
    n = x.shape[0] - (m - 1) * tau
    if n <= 0:
        return np.zeros((0, m), dtype=np.float64)
    return np.column_stack([x[i * tau : i * tau + n] for i in range(m)])


def _neighbour_indices(orbit: np.ndarray, theiler: int) -> np.ndarray:
    """Index of the nearest neighbour with |i-j| > theiler. -1 if none.

    A smooth trajectory's closest points are the temporal neighbours, so the
    query asks for more than ``2 * theiler`` neighbours and keeps the first
    one outside the window.
    """
    n = int(orbit.shape[0])
    theiler = int(max(0, theiler))
    k = min(n, 2 * theiler + 2)
    if k < 2:
        return np.full(n, -1, dtype=np.int64)
    dists, idxs = cKDTree(orbit).query(orbit, k=k)
    dists = np.atleast_2d(np.asarray(dists, dtype=np.float64))
    idxs = np.atleast_2d(np.asarray(idxs, dtype=np.int64))
    chosen = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        for dist, j in zip(dists[i], idxs[i]):
            j = int(j)
            if j < 0 or j >= n or j == i:
                continue
            if abs(j - i) <= theiler:
                continue
            if not np.isfinite(dist) or dist <= _DIST_FLOOR:
                continue
            chosen[i] = j
            break
    return chosen


def _linear_region_slope(log_div: np.ndarray, dt: float) -> float:
    """Slope of the settled scaling region, in 1/time.

    Early steps are a stretching transient (local slope above the exponent).
    The exponent is the least-squares slope on the longest run of local
    slopes that sit near the median of the later local slopes.
    """
    n = int(log_div.shape[0])
    if n < 4:
        return float("nan")
    window = max(8, n // 10)
    if window >= n - 2:
        time_axis = np.arange(n, dtype=np.float64) * float(dt)
        return float(np.polyfit(time_axis, log_div, 1)[0])
    local = np.empty(n - window, dtype=np.float64)
    span = float(window) * float(dt)
    for i in range(n - window):
        local[i] = (log_div[i + window] - log_div[i]) / span
    later = local[len(local) // 3 :]
    target = float(np.median(later))
    if not np.isfinite(target):
        return float("nan")
    tolerance = max(0.2, 0.4 * abs(target))
    close = np.abs(local - target) <= tolerance
    best_start, best_end = 0, 0
    i = 0
    while i < close.size:
        if not close[i]:
            i += 1
            continue
        j = i + 1
        while j < close.size and close[j]:
            j += 1
        if j - i > best_end - best_start:
            best_start, best_end = i, j
        i = j
    start = int(best_start)
    stop = min(n, int(best_end) + window)
    if stop - start < 4:
        start = n // 3
        stop = n
    time_axis = np.arange(start, stop, dtype=np.float64) * float(dt)
    slope = np.polyfit(time_axis, log_div[start:stop], 1)[0]
    if not np.isfinite(slope):
        return float("nan")
    return float(slope)


def rosenstein_lle(
    series: np.ndarray,
    *,
    m: int,
    tau: int,
    theiler: Optional[int] = None,
    dt: float = 1.0,
    evolution: int = DEFAULT_EVOLUTION,
    fit_steps: Optional[int] = None,
    fit_offset: int = 0,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> float:
    """Largest Lyapunov exponent in 1/time.

    ``theiler`` defaults to the embedding window ``(m - 1) * tau``.
    When ``fit_steps`` is omitted, the slope is taken on the settled linear
    region of the mean log-divergence curve (Rosenstein's scaling region),
    not the steeper first few steps. Returns NaN when the series is too
    short or too few pairs sit outside the Theiler window.
    """
    m = int(m)
    tau = int(tau)
    if theiler is None:
        theiler = max(1, (m - 1) * tau)
    theiler = int(theiler)
    evolution = int(evolution)
    fit_offset = int(fit_offset)
    if evolution < 2:
        return float("nan")
    if dt is None or not np.isfinite(dt) or float(dt) <= 0.0:
        dt = 1.0
    segment = longest_finite_segment(series)
    embedded = delay_embed(segment, m, tau)
    # Pairs are followed ``evolution`` steps inside the embedded series.
    n_orbit = int(embedded.shape[0]) - evolution
    if n_orbit < int(min_pairs):
        return float("nan")
    orbit = embedded[:n_orbit]
    neighbours = _neighbour_indices(orbit, theiler)
    valid = neighbours >= 0
    if int(valid.sum()) < int(min_pairs):
        return float("nan")
    ii = np.flatnonzero(valid)
    jj = neighbours[valid]
    if max_pairs is not None and ii.size > int(max_pairs):
        rng = np.random.default_rng(_PAIR_SAMPLE_SEED)
        take = rng.choice(ii.size, int(max_pairs), replace=False)
        ii = ii[take]
        jj = jj[take]
    log_div = np.empty(evolution, dtype=np.float64)
    for step in range(evolution):
        diff = embedded[ii + step] - embedded[jj + step]
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, _DIST_FLOOR)
        log_div[step] = float(np.mean(np.log(dist)))
    if fit_steps is None and fit_offset == 0:
        return _linear_region_slope(log_div, float(dt))
    start = max(0, fit_offset)
    n_fit = evolution - start if fit_steps is None else int(fit_steps)
    stop = start + n_fit
    if stop > evolution or n_fit < 2 or start >= evolution:
        return float("nan")
    time_axis = np.arange(start, stop, dtype=np.float64) * float(dt)
    slope = np.polyfit(time_axis, log_div[start:stop], 1)[0]
    if not np.isfinite(slope):
        return float("nan")
    return float(slope)


def _channel_embed_dim(benchmarks: Mapping[str, Any], key: str, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    return delay_dim_for_key(
        int(benchmarks.get("delay_dims") or 3),
        benchmarks.get("delay_dims_by_channel"),
        key,
    )


def rosenstein_on_benchmarks(
    channel_benchmarks: Mapping[str, Any],
    *,
    embed_dim: Optional[int] = None,
    tau: Optional[int] = None,
    theiler: Optional[int] = None,
    evolution: int = DEFAULT_EVOLUTION,
    fit_steps: Optional[int] = None,
    fit_offset: int = 0,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> dict[str, Any]:
    """GT and free-run Rosenstein exponents, plus Auto minus GT.

    Embedding dimension comes from ``embed_dim`` when set, otherwise from the
    delay-embedding choice already stored on the benchmarks (fixed config or
    the TwoNN / ID scan). Tau likewise. Theiler defaults to ``(m - 1) * tau``.
    """
    channels = list(channel_benchmarks.get("channels") or [])
    dt = float(channel_benchmarks.get("dt") or 1.0)
    default_tau = int(tau if tau is not None else channel_benchmarks.get("time_delay") or 1)
    per_channel: dict[str, dict[str, float]] = {}
    for ch in channels:
        key = str(ch["key"])
        m = _channel_embed_dim(channel_benchmarks, key, embed_dim)
        used_tau = default_tau
        used_theiler = int(theiler) if theiler is not None else max(1, (m - 1) * used_tau)
        gt = rosenstein_lle(
            ch["gt_auto"],
            m=m,
            tau=used_tau,
            theiler=used_theiler,
            dt=dt,
            evolution=evolution,
            fit_steps=fit_steps,
            fit_offset=fit_offset,
            min_pairs=min_pairs,
        )
        auto = rosenstein_lle(
            ch["auto_seg"],
            m=m,
            tau=used_tau,
            theiler=used_theiler,
            dt=dt,
            evolution=evolution,
            fit_steps=fit_steps,
            fit_offset=fit_offset,
            min_pairs=min_pairs,
        )
        diff = (
            float(auto - gt)
            if np.isfinite(auto) and np.isfinite(gt)
            else float("nan")
        )
        per_channel[key] = {
            "lle_rosenstein_gt": gt,
            "lle_rosenstein_auto": auto,
            "lle_rosenstein_diff": diff,
            "lle_rosenstein_embed_dim": float(m),
            "lle_rosenstein_tau": float(used_tau),
            "lle_rosenstein_theiler": float(used_theiler),
        }

    def _mean(field: str) -> float:
        vals = [
            float(row[field])
            for row in per_channel.values()
            if np.isfinite(row[field])
        ]
        return float(np.mean(vals)) if vals else float("nan")

    out: dict[str, Any] = {
        "per_channel": per_channel,
        "lle_rosenstein_gt": _mean("lle_rosenstein_gt"),
        "lle_rosenstein_auto": _mean("lle_rosenstein_auto"),
        "lle_rosenstein_diff": _mean("lle_rosenstein_diff"),
        "lle_rosenstein_dt": float(dt),
        "lle_rosenstein_evolution": int(evolution),
    }
    if per_channel:
        first = next(iter(per_channel.values()))
        out["lle_rosenstein_embed_dim"] = first["lle_rosenstein_embed_dim"]
        out["lle_rosenstein_tau"] = first["lle_rosenstein_tau"]
        out["lle_rosenstein_theiler"] = first["lle_rosenstein_theiler"]
    return out


def flatten_rosenstein(result: Mapping[str, Any]) -> dict[str, float]:
    """Scalar keys for one eval window. Settings stay out of the mean/std merge."""
    out: dict[str, float] = {}
    for key in (
        "lle_rosenstein_gt",
        "lle_rosenstein_auto",
        "lle_rosenstein_diff",
    ):
        if key in result and result[key] is not None:
            try:
                out[key] = float(result[key])
            except (TypeError, ValueError):
                continue
    for ch_key, vals in (result.get("per_channel") or {}).items():
        for prefix in (
            "lle_rosenstein_gt",
            "lle_rosenstein_auto",
            "lle_rosenstein_diff",
        ):
            if prefix in vals:
                try:
                    out[f"{prefix}_{ch_key}"] = float(vals[prefix])
                except (TypeError, ValueError):
                    continue
    return out
