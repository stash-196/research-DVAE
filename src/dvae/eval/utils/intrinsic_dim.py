"""Intrinsic-dimension estimators for evaluation clouds.

Participation ratio (PR) and the Facco et al. (2017) TwoNN estimator.
Both operate on arrays shaped (N, D) after rows that contain NaN are dropped.
Observation-space clouds are delay embeddings. Hidden-state joint ID uses the
model trajectory ``h`` directly (one point per time, all units together).
No new dependencies beyond numpy / scipy.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

# TwoNN linear fit drops the largest ratios, where 1 - F is noisy.
TWONN_DISCARD_FRACTION = 0.1
TWONN_MIN_POINTS = 20

# Plateau: smallest m whose next K relative ID steps stay under this.
DEFAULT_PLATEAU_EPS = 0.05
DEFAULT_PLATEAU_K = 2
DEFAULT_TWONN_M_MAX = 10

# Relative step |d(next)-d(m)| / max(|d(m)|, this floor).
PLATEAU_REL_FLOOR = 1e-8

# Eigenvalues smaller than this fraction of the largest are treated as numerical zero.
PR_EIGENVALUE_REL_TOL = 1e-10

# Summary keys written next to KLD. Per-channel keys are ``{prefix}_{channel}``.
ID_SUMMARY_KEYS: Tuple[str, ...] = (
    "id_pr_gt",
    "id_pr_tf",
    "id_pr_auto",
    "id_twonn_gt",
    "id_twonn_tf",
    "id_twonn_auto",
    "id_pr_joint_gt",
    "id_pr_joint_tf",
    "id_pr_joint_auto",
    "id_twonn_joint_gt",
    "id_twonn_joint_tf",
    "id_twonn_joint_auto",
    # Joint hidden state. No GT key: observations have no ground-truth latent.
    "id_pr_hidden_joint_tf",
    "id_pr_hidden_joint_auto",
    "id_twonn_hidden_joint_tf",
    "id_twonn_hidden_joint_auto",
)
# Stamped beside the scalars so evaluation_summary records why GT is absent.
ID_HIDDEN_JOINT_GT_NOTE = (
    "GT omitted: observations have no ground-truth latent. "
    "id_pr_hidden_joint_{tf,auto} and id_twonn_hidden_joint_{tf,auto} are "
    "PR and TwoNN on the model hidden state h (time, hidden units), batch 0, "
    "restricted to autonomous timesteps. TF is the all_0 teacher-forced pass; "
    "Auto is the free-run pass. LSTM cell state c is not included."
)
ID_PER_CHANNEL_PREFIXES: Tuple[str, ...] = (
    "id_pr_gt",
    "id_pr_tf",
    "id_pr_auto",
    "id_twonn_gt",
    "id_twonn_tf",
    "id_twonn_auto",
)

# Same preference as batch_all's primary channel (x / signal / ch4 / ch1).
PRIMARY_CHANNEL_PREFERENCE: Tuple[str, ...] = ("x", "signal", "ch4", "ch1")


def drop_nan_rows(X: np.ndarray) -> np.ndarray:
    """Return rows of ``X`` with no non-finite entries. Accepts a 1D series."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"expected (N, D) array, got shape {arr.shape}")
    if arr.shape[0] == 0:
        return arr
    finite = np.isfinite(arr).all(axis=1)
    return arr[finite]


def participation_ratio(X: np.ndarray) -> float:
    """Participation ratio of a point cloud.

    PR = (sum_i λ_i)^2 / sum_i λ_i^2 on the covariance eigenvalues of the
    centered cloud. An isotropic k-dimensional linear subspace has PR ≈ k.
    Rows with NaN are dropped. Returns NaN when fewer than 2 finite rows
    remain or the cloud has no variance.
    """
    arr = drop_nan_rows(X)
    n = arr.shape[0]
    if n < 2 or arr.shape[1] < 1:
        return float("nan")
    centered = arr - arr.mean(axis=0, keepdims=True)
    # Singular values of the centered matrix; λ_cov = s^2 / (n - 1).
    try:
        singular = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    lam = (singular ** 2) / float(n - 1)
    lam_max = float(np.max(lam)) if lam.size else 0.0
    if not np.isfinite(lam_max) or lam_max <= 0.0:
        return float("nan")
    lam = lam[lam > PR_EIGENVALUE_REL_TOL * lam_max]
    if lam.size == 0:
        return float("nan")
    total = float(np.sum(lam))
    denom = float(np.sum(lam ** 2))
    if denom <= 0.0:
        return float("nan")
    return float((total ** 2) / denom)


def twonn_id_from_ratios(
    mu: np.ndarray,
    discard_fraction: float = TWONN_DISCARD_FRACTION,
) -> float:
    """Facco TwoNN slope fit on ratios μ = r2 / r1.

    The empirical CDF satisfies 1 - F(μ) ≈ μ^(-d), so d is the through-origin
    slope of -log(1 - F) versus log(μ). The largest ``discard_fraction`` of
    ratios are left out of the fit (the far tail of F is unstable). The last
    ratio is always excluded because F = 1 there.
    """
    ratios = np.asarray(mu, dtype=np.float64).reshape(-1)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    if ratios.size < TWONN_MIN_POINTS:
        return float("nan")
    ratios = np.sort(ratios)
    n = int(ratios.size)
    keep = int(np.floor(n * (1.0 - float(discard_fraction))))
    # Need at least one point and must exclude F=1 at the final ratio.
    n_fit = max(3, min(keep, n - 1))
    if n_fit < 3:
        return float("nan")
    # F_i = i / N using the full sample size, matching the empirical CDF.
    empirical_f = np.arange(1, n_fit + 1, dtype=np.float64) / float(n)
    if np.any(empirical_f >= 1.0):
        return float("nan")
    log_mu = np.log(ratios[:n_fit])
    y = -np.log(1.0 - empirical_f)
    denom = float(np.dot(log_mu, log_mu))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    slope = float(np.dot(log_mu, y) / denom)
    if not np.isfinite(slope) or slope <= 0.0:
        return float("nan")
    return slope


def twonn_id(
    X: np.ndarray,
    discard_fraction: float = TWONN_DISCARD_FRACTION,
) -> float:
    """TwoNN intrinsic dimension (Facco et al. 2017).

    For each point, r1 and r2 are the distances to the first and second
    nearest neighbors (the query point itself is excluded). μ = r2 / r1.
    Points with a zero nearest-neighbor distance (duplicates) are dropped.
    """
    arr = drop_nan_rows(X)
    n = arr.shape[0]
    if n < TWONN_MIN_POINTS + 1:
        return float("nan")
    # k=3: self (distance 0), first neighbor, second neighbor.
    try:
        tree = cKDTree(arr)
        dists, _ = tree.query(arr, k=3)
    except Exception:
        return float("nan")
    dists = np.asarray(dists, dtype=np.float64)
    if dists.ndim != 2 or dists.shape[1] < 3:
        return float("nan")
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    good = np.isfinite(r1) & np.isfinite(r2) & (r1 > 0.0) & (r2 >= r1)
    if int(np.sum(good)) < TWONN_MIN_POINTS:
        return float("nan")
    return twonn_id_from_ratios(r2[good] / r1[good], discard_fraction=discard_fraction)


def _relative_step(current: float, nxt: float) -> float:
    return abs(float(nxt) - float(current)) / max(abs(float(current)), PLATEAU_REL_FLOOR)


def select_plateau_m(
    m_values: Sequence[int],
    id_values: Sequence[float],
    eps: float = DEFAULT_PLATEAU_EPS,
    k_consecutive: int = DEFAULT_PLATEAU_K,
) -> Tuple[Optional[int], bool]:
    """Smallest embedding dimension m* at which TwoNN ID plateaus.

    Plateau rule
    ------------
    Scan ``m_values`` in the order given (caller passes m = 1 .. m_max).
    Let d(m) be the TwoNN estimate. The step from m to the next scanned m
    is stable when both estimates are finite and

        |d(next) - d(m)| / max(|d(m)|, 1e-8) < eps

    m* is the smallest m such that the next ``k_consecutive`` steps are all
    stable. That m is the start of the plateau (the estimate has already
    stopped moving), not the end of it.

    If no such m exists, m* is the scanned m that is not the last entry and
    has the smallest relative step (ties: the smaller m). The boolean is
    then False. If fewer than two finite estimates exist, returns
    ``(None, False)``.
    """
    ms = [int(m) for m in m_values]
    ids = [float(v) for v in id_values]
    if len(ms) != len(ids):
        raise ValueError("m_values and id_values must have the same length")
    if k_consecutive < 1:
        raise ValueError("k_consecutive must be >= 1")
    if len(ms) < 2:
        return None, False

    finite_pairs = sum(
        1
        for i in range(len(ms) - 1)
        if np.isfinite(ids[i]) and np.isfinite(ids[i + 1])
    )
    if finite_pairs < 1:
        return None, False

    def _stable(i: int) -> bool:
        if not (np.isfinite(ids[i]) and np.isfinite(ids[i + 1])):
            return False
        return _relative_step(ids[i], ids[i + 1]) < float(eps)

    last_start = len(ms) - 1 - int(k_consecutive)
    for i in range(max(0, last_start + 1)):
        if all(_stable(i + offset) for offset in range(int(k_consecutive))):
            return ms[i], True

    best_i = None
    best_rel = None
    for i in range(len(ms) - 1):
        if not (np.isfinite(ids[i]) and np.isfinite(ids[i + 1])):
            continue
        rel = _relative_step(ids[i], ids[i + 1])
        if best_rel is None or rel < best_rel - 1e-15 or (
            abs(rel - best_rel) <= 1e-15 and (best_i is None or ms[i] < ms[best_i])
        ):
            best_rel = rel
            best_i = i
    if best_i is None:
        return None, False
    return ms[best_i], False


def id_metrics_for_clouds(
    gt_emb: Optional[np.ndarray],
    tf_emb: Optional[np.ndarray],
    auto_emb: Optional[np.ndarray],
) -> dict:
    """PR and TwoNN on the same three clouds used for per-channel DE→KLD."""

    def _one(emb: Optional[np.ndarray]) -> Tuple[float, float]:
        if emb is None:
            return float("nan"), float("nan")
        arr = np.asarray(emb, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return float("nan"), float("nan")
        return participation_ratio(arr), twonn_id(arr)

    pr_gt, tw_gt = _one(gt_emb)
    pr_tf, tw_tf = _one(tf_emb)
    pr_auto, tw_auto = _one(auto_emb)
    return {
        "id_pr_gt": pr_gt,
        "id_pr_tf": pr_tf,
        "id_pr_auto": pr_auto,
        "id_twonn_gt": tw_gt,
        "id_twonn_tf": tw_tf,
        "id_twonn_auto": tw_auto,
    }


def snapshot_hidden_state(model) -> Optional[np.ndarray]:
    """Copy ``model.h`` to a NumPy array.

    Returns None when the model has no hidden trajectory. Accepts a tensor
    (``detach`` / ``cpu`` / ``numpy``) or an array. Shape is left as stored,
    typically ``(time, batch, hidden)``.
    """
    h = getattr(model, "h", None)
    if h is None:
        return None
    if hasattr(h, "detach"):
        h = h.detach()
    if hasattr(h, "cpu"):
        h = h.cpu()
    if hasattr(h, "numpy"):
        h = h.numpy()
    arr = np.asarray(h)
    if arr.ndim not in (2, 3) or arr.size == 0:
        return None
    # Copy so a later forward that reuses the buffer cannot change the snapshot.
    return np.array(arr, dtype=np.float64, copy=True)


def hidden_state_cloud(
    h: Optional[np.ndarray],
    time_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Joint hidden cloud ``(N, H)`` for one trajectory.

    ``h`` may be ``(time, hidden)`` or ``(time, batch, hidden)``. The batch
    axis uses index 0, matching the hidden-state figures. ``time_mask``
    selects the same timesteps as the observation-space auto segment.
    """
    if h is None:
        return None
    arr = np.asarray(h, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    if time_mask is not None:
        mask = np.asarray(time_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != arr.shape[0] or not np.any(mask):
            return None
        arr = arr[mask]
    if arr.shape[0] < 2:
        return None
    return arr


def id_metrics_for_hidden_joint(
    h_tf: Optional[np.ndarray],
    h_auto: Optional[np.ndarray],
) -> dict:
    """PR and TwoNN on joint hidden trajectories.

    Each argument is already a ``(N, H)`` cloud (see ``hidden_state_cloud``).
    Ground truth is omitted: there is no comparable latent on the data side.
    Observation-space ``id_*_joint_*`` is a different family (delay embeddings).
    """
    metrics = id_metrics_for_clouds(None, h_tf, h_auto)
    return {
        "id_pr_hidden_joint_tf": metrics["id_pr_tf"],
        "id_pr_hidden_joint_auto": metrics["id_pr_auto"],
        "id_twonn_hidden_joint_tf": metrics["id_twonn_tf"],
        "id_twonn_hidden_joint_auto": metrics["id_twonn_auto"],
    }


def hidden_joint_id_from_benchmarks(channel_benchmarks: Optional[dict]) -> dict:
    """Hidden-joint ID from optional ``hidden_tf`` / ``hidden_auto`` arrays.

    Both clouds are masked with ``auto_mask`` when that mask is present, so
    TF and Auto are scored on the free-run timesteps. Returns an empty dict
    when neither trajectory was recorded. Includes ``id_hidden_joint_gt_note``
    (a string; numeric aggregators skip it).
    """
    if not channel_benchmarks:
        return {}
    h_tf = channel_benchmarks.get("hidden_tf")
    h_auto = channel_benchmarks.get("hidden_auto")
    if h_tf is None and h_auto is None:
        return {}
    mask = channel_benchmarks.get("auto_mask")
    metrics = id_metrics_for_hidden_joint(
        hidden_state_cloud(h_tf, mask),
        hidden_state_cloud(h_auto, mask),
    )
    metrics["id_hidden_joint_gt_note"] = ID_HIDDEN_JOINT_GT_NOTE
    return metrics


def primary_channel_key(keys: Sequence[str]) -> str:
    """Channel whose scalar delay dimension drives stitched KLD / batch_all."""
    present = [str(k) for k in keys]
    for prefer in PRIMARY_CHANNEL_PREFERENCE:
        if prefer in present:
            return prefer
    if not present:
        raise ValueError("primary_channel_key requires at least one channel")
    return present[0]


def delay_dim_for_key(
    delay_dims: int,
    delay_dims_by_channel: Optional[dict],
    key: str,
) -> int:
    """Per-channel m* when the scan stored one, otherwise the shared scalar."""
    by_ch = delay_dims_by_channel or {}
    if key in by_ch and by_ch[key] is not None:
        return int(by_ch[key])
    return int(delay_dims)
