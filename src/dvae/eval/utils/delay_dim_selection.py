"""Choose delay-embedding (tau, m) for geometry eval.

Default (``fixed``) keeps ``_get_delay_params`` so existing evals replicate.
``twonn_scan`` estimates m* from ground-truth auto segments only, writes
``gt_delay_embed_params.yaml`` on the experiment root, and later runs under
the same experiment load that file instead of scanning again.

Tau is never chosen here. It stays at ``_get_delay_params`` or the CLI
override, and a cache hit also freezes the tau that was scanned.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import yaml

from dvae.eval.utils.delay_embedding import compute_delay_embedding
from dvae.eval.utils.intrinsic_dim import (
    DEFAULT_PLATEAU_EPS,
    DEFAULT_PLATEAU_K,
    DEFAULT_TWONN_M_MAX,
    participation_ratio,
    primary_channel_key,
    select_plateau_m,
    twonn_id,
)

CACHE_FILENAME = "gt_delay_embed_params.yaml"
METHOD_FIXED = "fixed"
METHOD_TWONN_SCAN = "twonn_scan"

# Directories that sit next to run folders but are not runs.
_SKIP_DIR_NAMES = {
    "logs",
    "resume_logs",
    "eval_logs",
    "temp",
    "post_training_figs",
    "metrics_batches",
    "batch_all",
}

GtLoader = Callable[[], Dict[str, List[np.ndarray]]]


@dataclass
class DelayEmbedChoice:
    """Embedding parameters applied to GT, TF, and Auto together."""

    method: str
    time_delay: int
    delay_dims: int
    delay_dims_by_channel: Dict[str, int] = field(default_factory=dict)
    joint_delay_dims: Optional[int] = None
    source: str = METHOD_FIXED
    cache_path: Optional[str] = None
    fingerprint: Optional[dict] = None
    fingerprint_match: Optional[bool] = None
    plateau_found: Dict[str, bool] = field(default_factory=dict)
    primary_channel: Optional[str] = None

    def as_benchmark_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``get_channel_benchmarks``."""
        kwargs: Dict[str, Any] = {
            "time_delay": int(self.time_delay),
            "delay_dims": int(self.delay_dims),
        }
        if self.delay_dims_by_channel:
            kwargs["delay_dims_by_channel"] = {
                str(k): int(v) for k, v in self.delay_dims_by_channel.items()
            }
        if self.joint_delay_dims is not None:
            kwargs["joint_delay_dims"] = int(self.joint_delay_dims)
        return kwargs


def _default_delay_params(dataset_name: str):
    from dvae.eval.utils.delay_params import _get_delay_params

    return _get_delay_params(dataset_name)


def _is_run_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "config.ini")):
        return True
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return any(name.endswith(".pt") for name in names)


def _run_child_dirs(path: str) -> List[str]:
    try:
        names = os.listdir(path)
    except OSError:
        return []
    children = []
    for name in names:
        if name in _SKIP_DIR_NAMES or name.startswith("aggregate_eval_plots"):
            continue
        if name.startswith("."):
            continue
        child = os.path.join(path, name)
        if os.path.isdir(child) and _is_run_dir(child):
            children.append(child)
    return children


def resolve_experiment_root(run_dir: str) -> str:
    """Directory that owns the per-run folders of one sweep.

    Eval is pointed at a run directory (``config.ini`` and weights live
    there). ``run_eval_multiple.sh`` treats the experiment as the parent of
    those run directories (``find -mindepth 1 -maxdepth 1``). The GT delay
    cache is written once on that parent so sibling runs reuse it.

    If ``run_dir`` is already that parent (it contains run children and is
    not itself a run), it is returned unchanged.
    """
    path = os.path.abspath(run_dir)
    if _is_run_dir(path):
        return os.path.dirname(path)
    if _run_child_dirs(path):
        return path
    cur = path
    for _ in range(4):
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        if _is_run_dir(parent):
            return os.path.dirname(parent)
        if _run_child_dirs(parent):
            return parent
        cur = parent
    return os.path.dirname(path)


def default_cache_path(run_dir: str) -> str:
    return os.path.join(resolve_experiment_root(run_dir), CACHE_FILENAME)


def make_gt_fingerprint(
    *,
    dataset_name: str,
    seq_len: int,
    n_windows: int,
    auto_eval_mode: str,
    time_delay: int = 0,
    m_max: int = 0,
    observation_process: Optional[str] = None,
    auto_eval_block_len: Optional[int] = None,
    auto_eval_ratio: Optional[float] = None,
    auto_eval_flip_point: Optional[int] = None,
) -> dict:
    """How the GT scan was defined. Not a hash of the sample values.

    Stored so a later eval can see that seq length / window count / schedule
    no longer match the cache. A mismatch is reported; it does not by itself
    trigger a rescan (use ``--force-gt-delay-rescan`` for that).
    """
    return {
        "dataset_name": str(dataset_name),
        "seq_len": int(seq_len),
        "n_windows": int(n_windows),
        "auto_eval_mode": str(auto_eval_mode),
        "observation_process": "" if observation_process is None else str(observation_process),
        "auto_eval_block_len": (
            None if auto_eval_block_len is None else int(auto_eval_block_len)
        ),
        "auto_eval_ratio": (
            None if auto_eval_ratio is None else float(auto_eval_ratio)
        ),
        "auto_eval_flip_point": (
            None if auto_eval_flip_point is None else int(auto_eval_flip_point)
        ),
        "time_delay": int(time_delay),
        "m_max": int(m_max),
    }


def fingerprint_hash(fingerprint: Optional[dict]) -> Optional[str]:
    if not fingerprint:
        return None
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _yaml_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _yaml_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _yaml_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_gt_delay_cache(path: str, payload: dict) -> None:
    """Write the cache atomically (temp file in the same directory, then rename)."""
    path = os.path.abspath(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".gt_delay_embed_", suffix=".yaml.tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                _yaml_ready(payload),
                handle,
                sort_keys=False,
                default_flow_style=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_gt_delay_cache(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"GT delay cache is not a mapping: {path}")
    return data


def _embed_segment(segment: np.ndarray, tau: int, m: int) -> Optional[np.ndarray]:
    sig = np.asarray(segment, dtype=np.float64).reshape(-1)
    if sig.size < (m - 1) * tau + 2:
        return None
    try:
        embedded = compute_delay_embedding(
            sig, delay=int(tau), dimensions=int(m), handle_nan="remove"
        )
    except ValueError:
        return None
    embedded = np.asarray(embedded, dtype=np.float64)
    if embedded.ndim != 2 or embedded.shape[0] == 0:
        return None
    return embedded


def pool_delay_embeddings(
    segments: Sequence[np.ndarray],
    tau: int,
    m: int,
) -> Optional[np.ndarray]:
    """Embed each window on its own, then stack rows (no bridge across gaps)."""
    rows: List[np.ndarray] = []
    for segment in segments:
        embedded = _embed_segment(segment, tau, m)
        if embedded is not None:
            rows.append(embedded)
    if not rows:
        return None
    return np.vstack(rows)


def _curve_for_segments(
    segments: Sequence[np.ndarray],
    tau: int,
    m_max: int,
) -> Dict[str, List[Optional[float]]]:
    ms: List[int] = []
    twonn_vals: List[Optional[float]] = []
    pr_vals: List[Optional[float]] = []
    for m in range(1, int(m_max) + 1):
        cloud = pool_delay_embeddings(segments, tau, m)
        ms.append(m)
        if cloud is None:
            twonn_vals.append(None)
            pr_vals.append(None)
            continue
        pr = participation_ratio(cloud)
        tw = twonn_id(cloud)
        pr_vals.append(None if not np.isfinite(pr) else float(pr))
        twonn_vals.append(None if not np.isfinite(tw) else float(tw))
    return {"m": ms, "twonn": twonn_vals, "pr": pr_vals}


def _joint_segments_as_clouds(
    segments_by_channel: Dict[str, List[np.ndarray]],
    tau: int,
    m: int,
) -> Optional[np.ndarray]:
    """Per-window horizontal stack of per-channel embeddings, then row-stack."""
    keys = list(segments_by_channel.keys())
    if len(keys) < 2:
        return None
    n_windows = min(len(segments_by_channel[k]) for k in keys)
    rows: List[np.ndarray] = []
    for index in range(n_windows):
        embs = []
        for key in keys:
            emb = _embed_segment(segments_by_channel[key][index], tau, m)
            if emb is None:
                embs = []
                break
            embs.append(emb)
        if not embs:
            continue
        if any(emb.shape[0] != embs[0].shape[0] for emb in embs):
            continue
        rows.append(np.hstack(embs))
    if not rows:
        return None
    return np.vstack(rows)


def scan_gt_delay_dims(
    segments_by_channel: Dict[str, List[np.ndarray]],
    tau: int,
    m_max: int = DEFAULT_TWONN_M_MAX,
    eps: float = DEFAULT_PLATEAU_EPS,
    k_consecutive: int = DEFAULT_PLATEAU_K,
    fallback_delay_dims: int = 3,
) -> dict:
    """TwoNN-vs-m on GT only. Tau is fixed.

    Each channel is scanned separately. When more than one channel is
    present, a joint curve stacks per-channel embeddings at the same m
    (diagnostic; it does not replace per-channel DE→KLD).
    """
    if int(m_max) < 1:
        raise ValueError(f"m_max must be >= 1, got {m_max}")
    if int(tau) < 1:
        raise ValueError(f"time_delay must be >= 1, got {tau}")

    curves: Dict[str, dict] = {}
    selected: Dict[str, int] = {}
    plateau_found: Dict[str, bool] = {}

    for key, segments in segments_by_channel.items():
        curve = _curve_for_segments(segments, tau, m_max)
        curves[str(key)] = curve
        twonn_for_picker = [
            float("nan") if v is None else float(v) for v in curve["twonn"]
        ]
        m_star, found = select_plateau_m(
            curve["m"], twonn_for_picker, eps=eps, k_consecutive=k_consecutive
        )
        if m_star is None:
            m_star = int(fallback_delay_dims)
            found = False
        selected[str(key)] = int(m_star)
        plateau_found[str(key)] = bool(found)

    joint_curve = None
    joint_m = None
    joint_found = None
    if len(segments_by_channel) > 1:
        joint_ms = list(range(1, int(m_max) + 1))
        joint_tw: List[Optional[float]] = []
        joint_pr: List[Optional[float]] = []
        for m in joint_ms:
            cloud = _joint_segments_as_clouds(segments_by_channel, tau, m)
            if cloud is None:
                joint_tw.append(None)
                joint_pr.append(None)
                continue
            pr = participation_ratio(cloud)
            tw = twonn_id(cloud)
            joint_pr.append(None if not np.isfinite(pr) else float(pr))
            joint_tw.append(None if not np.isfinite(tw) else float(tw))
        joint_curve = {"m": joint_ms, "twonn": joint_tw, "pr": joint_pr}
        twonn_for_picker = [float("nan") if v is None else float(v) for v in joint_tw]
        joint_m, joint_found = select_plateau_m(
            joint_ms, twonn_for_picker, eps=eps, k_consecutive=k_consecutive
        )
        if joint_m is None:
            joint_m = int(fallback_delay_dims)
            joint_found = False
        curves["joint"] = joint_curve

    keys = list(selected.keys())
    primary = primary_channel_key(keys) if keys else None
    scalar_m = int(selected[primary]) if primary is not None else int(fallback_delay_dims)
    return {
        "delay_dims_selected": selected,
        "plateau_found": plateau_found,
        "curves": curves,
        "joint_delay_dims": None if joint_m is None else int(joint_m),
        "joint_plateau_found": joint_found,
        "primary_channel": primary,
        "delay_dims": scalar_m,
        "eps": float(eps),
        "k_consecutive": int(k_consecutive),
    }


def _fixed_choice(
    dataset_name: str,
    time_delay_override: Optional[int],
    delay_dims_override: Optional[int],
    source: str = METHOD_FIXED,
) -> DelayEmbedChoice:
    tau, delay_dims = _default_delay_params(dataset_name)
    if time_delay_override is not None:
        tau = int(time_delay_override)
    if delay_dims_override is not None:
        delay_dims = int(delay_dims_override)
    if int(tau) < 1 or int(delay_dims) < 1:
        raise ValueError(
            f"time_delay and delay_dims must be >= 1, got tau={tau}, m={delay_dims}"
        )
    return DelayEmbedChoice(
        method=METHOD_FIXED if source == METHOD_FIXED else METHOD_TWONN_SCAN,
        time_delay=int(tau),
        delay_dims=int(delay_dims),
        source=source,
    )


def _choice_from_cache(
    data: dict,
    cache_path: str,
    fingerprint: Optional[dict],
) -> DelayEmbedChoice:
    selected_raw = data.get("delay_dims_selected") or {}
    if not isinstance(selected_raw, dict) or not selected_raw:
        raise ValueError("cache is missing delay_dims_selected")
    selected = {str(k): int(v) for k, v in selected_raw.items()}
    primary = data.get("primary_channel") or primary_channel_key(list(selected))
    if primary not in selected:
        primary = primary_channel_key(list(selected))
    tau = int(data["time_delay"])
    delay_dims = int(data.get("delay_dims", selected[primary]))
    joint = data.get("joint_delay_dims")
    stored_fp = data.get("fingerprint")
    match = None
    if fingerprint is not None and data.get("fingerprint_hash"):
        match = fingerprint_hash(fingerprint) == data.get("fingerprint_hash")
    plateau = data.get("plateau_found") or {}
    return DelayEmbedChoice(
        method=METHOD_TWONN_SCAN,
        time_delay=tau,
        delay_dims=delay_dims,
        delay_dims_by_channel=selected,
        joint_delay_dims=None if joint is None else int(joint),
        source="cache",
        cache_path=os.path.abspath(cache_path),
        fingerprint=stored_fp if isinstance(stored_fp, dict) else fingerprint,
        fingerprint_match=match,
        plateau_found={str(k): bool(v) for k, v in plateau.items()} if isinstance(plateau, dict) else {},
        primary_channel=str(primary),
    )


def _gt_auto_points(segments_by_channel: Dict[str, List[np.ndarray]]) -> Dict[str, int]:
    points = {}
    for key, segments in segments_by_channel.items():
        points[str(key)] = int(sum(int(np.asarray(seg).reshape(-1).size) for seg in segments))
    return points


def resolve_delay_embedding(
    dataset_name: str,
    *,
    method: str = METHOD_FIXED,
    run_dir: Optional[str] = None,
    time_delay_override: Optional[int] = None,
    delay_dims_override: Optional[int] = None,
    m_max: int = DEFAULT_TWONN_M_MAX,
    cache_path: Optional[str] = None,
    force_rescan: bool = False,
    fingerprint: Optional[dict] = None,
    gt_segments_loader: Optional[GtLoader] = None,
    eps: float = DEFAULT_PLATEAU_EPS,
    k_consecutive: int = DEFAULT_PLATEAU_K,
) -> DelayEmbedChoice:
    """Resolve (tau, m) for one eval.

    ``fixed`` (default) returns ``_get_delay_params`` with optional integer
    overrides and does not read or write the cache.

    ``twonn_scan`` loads ``gt_delay_embed_params.yaml`` when it already
    exists and ``force_rescan`` is false. Otherwise it calls
    ``gt_segments_loader`` once, scans GT, and writes the cache.
    """
    method_name = str(method or METHOD_FIXED)
    if method_name not in (METHOD_FIXED, METHOD_TWONN_SCAN):
        raise ValueError(
            f"delay_dim_method must be '{METHOD_FIXED}' or '{METHOD_TWONN_SCAN}', "
            f"got {method_name!r}"
        )
    if method_name == METHOD_FIXED:
        return _fixed_choice(dataset_name, time_delay_override, delay_dims_override)

    fixed = _fixed_choice(dataset_name, time_delay_override, delay_dims_override)
    tau = int(fixed.time_delay)
    fallback_m = int(fixed.delay_dims)
    if fingerprint is None:
        fingerprint = {}
    else:
        fingerprint = dict(fingerprint)
    # Always record the tau / m_max this run would scan with, so a cache
    # written here matches a later load that uses the same overrides.
    fingerprint["dataset_name"] = str(dataset_name)
    fingerprint["time_delay"] = tau
    fingerprint["m_max"] = int(m_max)

    if cache_path:
        resolved_cache = os.path.abspath(cache_path)
    elif run_dir:
        resolved_cache = default_cache_path(run_dir)
    else:
        raise ValueError("twonn_scan requires run_dir or cache_path")

    if os.path.isfile(resolved_cache) and not force_rescan:
        try:
            loaded = load_gt_delay_cache(resolved_cache)
            choice = _choice_from_cache(loaded, resolved_cache, fingerprint)
            if choice.fingerprint_match is False:
                print(
                    "[Eval][DelayDim] GT delay cache fingerprint differs from this "
                    f"eval ({resolved_cache}). Using cached m* and tau anyway. "
                    "Pass --force-gt-delay-rescan to recompute."
                )
            else:
                print(
                    f"[Eval][DelayDim] Loaded GT delay cache {resolved_cache} "
                    f"(tau={choice.time_delay}, m*={choice.delay_dims_by_channel})"
                )
            return choice
        except Exception as exc:
            print(
                f"[Eval][DelayDim] Could not read cache {resolved_cache} ({exc}); "
                "rescanning GT."
            )

    if gt_segments_loader is None:
        raise ValueError("twonn_scan needs gt_segments_loader when the cache is missing")

    print(
        f"[Eval][DelayDim] TwoNN-vs-m scan on GT (tau={tau}, m_max={m_max}). "
        f"Cache: {resolved_cache}"
    )
    segments = gt_segments_loader()
    n_points = sum(_gt_auto_points(segments).values()) if segments else 0
    if n_points <= 0:
        # Do not cache this. An empty auto mask (for example all-TF) would
        # otherwise freeze the fallback m* for every later run in the sweep.
        print(
            "[Eval][DelayDim] No GT auto-segment samples; "
            "falling back to fixed delay params without writing a cache."
        )
        fallback = _fixed_choice(
            dataset_name,
            time_delay_override,
            delay_dims_override,
            source="fallback_fixed",
        )
        fallback.cache_path = resolved_cache
        fallback.fingerprint = fingerprint
        return fallback

    scan = scan_gt_delay_dims(
        segments,
        tau=tau,
        m_max=int(m_max),
        eps=eps,
        k_consecutive=k_consecutive,
        fallback_delay_dims=fallback_m,
    )
    payload = {
        "method": METHOD_TWONN_SCAN,
        "dataset_name": str(dataset_name),
        "time_delay": tau,
        "m_max": int(m_max),
        "plateau_eps": float(eps),
        "plateau_k_consecutive": int(k_consecutive),
        "plateau_rule": (
            "m* is the smallest m such that the next k_consecutive relative "
            "TwoNN steps |d(next)-d(m)|/max(|d(m)|, 1e-8) are all < eps. "
            "If none, m* is the m with the smallest relative step."
        ),
        "delay_dims_selected": scan["delay_dims_selected"],
        "delay_dims": scan["delay_dims"],
        "primary_channel": scan["primary_channel"],
        "joint_delay_dims": scan["joint_delay_dims"],
        "joint_plateau_found": scan["joint_plateau_found"],
        "plateau_found": scan["plateau_found"],
        "curves": scan["curves"],
        "fingerprint": fingerprint,
        "fingerprint_hash": fingerprint_hash(fingerprint),
        "gt_auto_points": _gt_auto_points(segments),
    }
    save_gt_delay_cache(resolved_cache, payload)
    print(
        f"[Eval][DelayDim] Wrote {resolved_cache} "
        f"m*={scan['delay_dims_selected']} joint_m={scan['joint_delay_dims']}"
    )
    return DelayEmbedChoice(
        method=METHOD_TWONN_SCAN,
        time_delay=tau,
        delay_dims=int(scan["delay_dims"]),
        delay_dims_by_channel={
            str(k): int(v) for k, v in scan["delay_dims_selected"].items()
        },
        joint_delay_dims=scan["joint_delay_dims"],
        source="computed",
        cache_path=resolved_cache,
        fingerprint=fingerprint,
        fingerprint_match=True,
        plateau_found=dict(scan["plateau_found"]),
        primary_channel=scan["primary_channel"],
    )


def delay_choice_summary_fields(choice: DelayEmbedChoice) -> Dict[str, Any]:
    """Scalars / small dicts stamped into ``evaluation_summary.yaml``."""
    fields: Dict[str, Any] = {
        "delay_dim_method": choice.method,
        "time_delay": int(choice.time_delay),
        "delay_dims": int(choice.delay_dims),
        "delay_dims_source": choice.source,
    }
    if choice.method == METHOD_TWONN_SCAN:
        fields["delay_dims_selected"] = {
            str(k): int(v) for k, v in choice.delay_dims_by_channel.items()
        }
        fields["gt_delay_cache"] = choice.cache_path
        if choice.joint_delay_dims is not None:
            fields["joint_delay_dims"] = int(choice.joint_delay_dims)
        if choice.primary_channel is not None:
            fields["delay_dims_primary_channel"] = choice.primary_channel
        if choice.fingerprint is not None:
            fields["gt_delay_fingerprint"] = choice.fingerprint
        if choice.fingerprint_match is not None:
            fields["gt_delay_fingerprint_match"] = bool(choice.fingerprint_match)
        if choice.plateau_found:
            fields["delay_dim_plateau_found"] = {
                str(k): bool(v) for k, v in choice.plateau_found.items()
            }
    return fields


def collect_gt_auto_segments(
    dataloader,
    *,
    dataset,
    dataset_name: str,
    observation_process: str,
    auto_mode: str,
    max_windows: int,
    block_len: Optional[int] = None,
    autonomous_ratio: float = 0.0,
    flip_point: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, List[np.ndarray]]:
    """GT auto-segment series for the same windows geometry will score.

    Reconstructions are zeros: only ``gt_auto`` is kept. The auto mask follows
    ``auto_mode`` (the same 1D schedule the metrics forward uses), so the scan
    does not depend on model weights.
    """
    from dvae.eval.utils.benchmark_signals import get_channel_benchmarks
    from dvae.eval.utils.forward_modes import get_flip_point_for_mode

    segments: Dict[str, List[np.ndarray]] = {}
    for i, batch in enumerate(dataloader):
        if i >= int(max_windows):
            break
        batch = batch.to(device).permute(1, 0, 2)
        seq_len = int(batch.shape[0])
        x_dim = int(batch.shape[2])
        try:
            window_flip = get_flip_point_for_mode(
                seq_len,
                auto_mode,
                flip_point=flip_point,
                block_len=block_len,
            )
        except ValueError:
            window_flip = seq_len // 2
        dummy = np.zeros((seq_len, x_dim), dtype=np.float64)
        benchmarks = get_channel_benchmarks(
            batch_data_long=batch,
            recon_tf=dummy,
            recon_auto_warmed=dummy,
            flip_point=window_flip,
            dataset=dataset,
            batch_idx=i,
            observation_process=observation_process or "",
            dataset_name=dataset_name,
            auto_mode=auto_mode,
            mode_selector=None,
            block_len=block_len,
            autonomous_ratio=autonomous_ratio,
        )
        for channel in benchmarks["channels"]:
            segments.setdefault(channel["key"], []).append(
                np.asarray(channel["gt_auto"], dtype=np.float64).reshape(-1).copy()
            )
    return segments
