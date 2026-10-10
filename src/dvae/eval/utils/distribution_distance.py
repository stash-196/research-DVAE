"""Earth-mover distances between free-run and ground-truth clouds.

Per observed dimension: 1D Wasserstein-1 (scipy). Joint cloud: sliced
Wasserstein (mean of 1D W1 on random projections). Exact multi-dimensional
EMD uses POT (``ot.emd2``) only when the subsampled cloud is small enough
to be cheap. POT is optional; when it is missing or the cloud is large the
exact score is NaN and a skip reason is recorded.

Samples are subsampled. Non-finite rows are dropped. No Numba.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
from scipy.stats import wasserstein_distance

DEFAULT_EMD_MAX_SAMPLES = 512
DEFAULT_EMD_N_PROJECTIONS = 32
DEFAULT_EMD_SEED = 0
# Exact POT below these caps. n=200 in dim<=5 is a few hundred milliseconds.
DEFAULT_EMD_EXACT_MAX_N = 200
DEFAULT_EMD_EXACT_MAX_DIM = 5


def _finite_rows(cloud: np.ndarray) -> np.ndarray:
    arr = np.asarray(cloud, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, arr.shape[1] if arr.ndim == 2 else 0), dtype=np.float64)
    return arr[np.isfinite(arr).all(axis=1)]


def _subsample_rows(cloud: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_samples is None or cloud.shape[0] <= int(max_samples):
        return cloud
    idx = rng.choice(cloud.shape[0], int(max_samples), replace=False)
    return cloud[idx]


def wasserstein_1d(a: np.ndarray, b: np.ndarray, *, max_samples: int, rng: np.random.Generator) -> float:
    """Empirical W1 between two 1D samples. NaN if either side is empty."""
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return float("nan")
    left = _subsample_rows(left.reshape(-1, 1), max_samples, rng).reshape(-1)
    right = _subsample_rows(right.reshape(-1, 1), max_samples, rng).reshape(-1)
    return float(wasserstein_distance(left, right))


def per_dimension_w1(
    reference: np.ndarray,
    other: np.ndarray,
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """W1 of each column. Length is the shared column count."""
    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(other, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    n_dim = int(min(ref.shape[1], pred.shape[1]))
    out = np.full(n_dim, np.nan, dtype=np.float64)
    for dim in range(n_dim):
        out[dim] = wasserstein_1d(
            ref[:, dim], pred[:, dim], max_samples=max_samples, rng=rng
        )
    return out


def sliced_wasserstein(
    reference: np.ndarray,
    other: np.ndarray,
    *,
    n_projections: int,
    max_samples: int,
    rng: np.random.Generator,
) -> float:
    """Mean 1D W1 over random unit directions. NaN if a cloud is empty."""
    ref = _subsample_rows(_finite_rows(reference), max_samples, rng)
    pred = _subsample_rows(_finite_rows(other), max_samples, rng)
    if ref.shape[0] == 0 or pred.shape[0] == 0:
        return float("nan")
    dim = int(min(ref.shape[1], pred.shape[1]))
    if dim < 1:
        return float("nan")
    ref = ref[:, :dim]
    pred = pred[:, :dim]
    n_projections = int(max(1, n_projections))
    directions = rng.normal(size=(n_projections, dim))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    directions = directions / norms
    scores = np.empty(n_projections, dtype=np.float64)
    for i, direction in enumerate(directions):
        scores[i] = float(wasserstein_distance(ref @ direction, pred @ direction))
    return float(np.mean(scores))


def exact_emd(
    reference: np.ndarray,
    other: np.ndarray,
    *,
    max_n: int,
    max_dim: int,
) -> tuple[float, Optional[str]]:
    """POT earth-mover distance, or (NaN, reason) when that would not be cheap.

    ``max_n`` / ``max_dim`` are the cheap-problem caps. Clouds larger than
    ``max_n`` are not solved; they are skipped rather than silently truncated
    into a different problem, except that a cloud already at most ``max_n``
    is used as-is.
    """
    ref = _finite_rows(reference)
    pred = _finite_rows(other)
    if ref.shape[0] == 0 or pred.shape[0] == 0:
        return float("nan"), "empty"
    dim = int(min(ref.shape[1], pred.shape[1]))
    if dim < 1:
        return float("nan"), "empty"
    if dim > int(max_dim):
        return float("nan"), "dim_above_cap"
    if ref.shape[0] > int(max_n) or pred.shape[0] > int(max_n):
        return float("nan"), "n_above_cap"
    ref = ref[:, :dim]
    pred = pred[:, :dim]
    try:
        import ot
    except ImportError:
        return float("nan"), "pot_unavailable"
    a = np.full(ref.shape[0], 1.0 / ref.shape[0])
    b = np.full(pred.shape[0], 1.0 / pred.shape[0])
    cost = ot.dist(ref, pred, metric="euclidean")
    value = float(ot.emd2(a, b, cost))
    if not np.isfinite(value):
        return float("nan"), "non_finite"
    return value, None


def _stack_channel_field(channels: list, field: str) -> np.ndarray:
    cols = []
    for ch in channels:
        series = np.asarray(ch[field], dtype=np.float64).reshape(-1)
        cols.append(series)
    if not cols:
        return np.zeros((0, 0), dtype=np.float64)
    length = min(col.shape[0] for col in cols)
    if length <= 0:
        return np.zeros((0, len(cols)), dtype=np.float64)
    return np.column_stack([col[:length] for col in cols])


def distribution_distances_on_benchmarks(
    channel_benchmarks: Mapping[str, Any],
    *,
    max_samples: int = DEFAULT_EMD_MAX_SAMPLES,
    n_projections: int = DEFAULT_EMD_N_PROJECTIONS,
    seed: int = DEFAULT_EMD_SEED,
    exact_max_n: int = DEFAULT_EMD_EXACT_MAX_N,
    exact_max_dim: int = DEFAULT_EMD_EXACT_MAX_DIM,
) -> dict[str, Any]:
    """1D W1, sliced Wasserstein, and optional exact EMD for TF and free-run.

    Free-run is the autonomous segment (``auto_seg``) against ``gt_auto``.
    Teacher-forced uses the same timesteps (``tf_auto``) so the two scores
    share a ground-truth distribution.
    """
    channels = list(channel_benchmarks.get("channels") or [])
    gt = _stack_channel_field(channels, "gt_auto")
    tf = _stack_channel_field(channels, "tf_auto")
    auto = _stack_channel_field(channels, "auto_seg")
    rng = np.random.default_rng(int(seed))

    def _one(other: np.ndarray, tag: str) -> dict[str, Any]:
        # Independent streams so TF and Auto subsamples do not share draws.
        local = np.random.default_rng(rng.integers(0, 2**31 - 1))
        w1 = per_dimension_w1(gt, other, max_samples=max_samples, rng=local)
        finite = w1[np.isfinite(w1)]
        sliced = sliced_wasserstein(
            gt,
            other,
            n_projections=n_projections,
            max_samples=max_samples,
            rng=local,
        )
        exact, reason = exact_emd(
            gt,
            other,
            max_n=exact_max_n,
            max_dim=exact_max_dim,
        )
        per_dim = {}
        for idx, ch in enumerate(channels):
            if idx < w1.shape[0]:
                per_dim[str(ch["key"])] = float(w1[idx])
        return {
            f"emd_w1_{tag}": float(np.mean(finite)) if finite.size else float("nan"),
            f"emd_sliced_{tag}": float(sliced),
            f"emd_exact_{tag}": float(exact),
            f"emd_exact_skip_reason_{tag}": reason,
            f"per_dim_{tag}": per_dim,
        }

    tf_scores = _one(tf, "tf")
    auto_scores = _one(auto, "auto")
    out: dict[str, Any] = {}
    out.update(tf_scores)
    out.update(auto_scores)
    out["emd_max_samples"] = int(max_samples)
    out["emd_n_projections"] = int(n_projections)
    out["emd_exact_max_n"] = int(exact_max_n)
    out["emd_exact_max_dim"] = int(exact_max_dim)
    return out


def flatten_distribution_distances(result: Mapping[str, Any]) -> dict[str, float]:
    """Scalar scores for one window. Skip reasons and caps are not included."""
    out: dict[str, float] = {}
    for key in (
        "emd_w1_tf",
        "emd_w1_auto",
        "emd_sliced_tf",
        "emd_sliced_auto",
        "emd_exact_tf",
        "emd_exact_auto",
    ):
        if key in result and result[key] is not None:
            try:
                out[key] = float(result[key])
            except (TypeError, ValueError):
                continue
    for tag in ("tf", "auto"):
        for ch_key, value in (result.get(f"per_dim_{tag}") or {}).items():
            try:
                out[f"emd_w1_{tag}_{ch_key}"] = float(value)
            except (TypeError, ValueError):
                continue
    return out
