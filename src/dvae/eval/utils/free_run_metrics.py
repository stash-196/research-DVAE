"""Wire EMD and Rosenstein LLE into one eval window.

Defaults match the module constants. An empty or missing ``[Evaluation]``
key keeps that default, so an existing config that never mentions these
options still records the scores.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from dvae.eval.utils.distribution_distance import (
    DEFAULT_EMD_EXACT_MAX_DIM,
    DEFAULT_EMD_EXACT_MAX_N,
    DEFAULT_EMD_MAX_SAMPLES,
    DEFAULT_EMD_N_PROJECTIONS,
    DEFAULT_EMD_SEED,
    distribution_distances_on_benchmarks,
    flatten_distribution_distances,
)
from dvae.eval.utils.rosenstein_lyapunov import (
    DEFAULT_EVOLUTION,
    flatten_rosenstein,
    rosenstein_on_benchmarks,
)

_UNSET = {"", "none", "null"}


def _raw_option(cfg, option: str) -> Optional[str]:
    if cfg is None:
        return None
    if not cfg.has_section("Evaluation") or not cfg.has_option("Evaluation", option):
        return None
    text = str(cfg.get("Evaluation", option)).strip()
    if text.lower() in _UNSET:
        return None
    return text


def _int_option(cfg, option: str, default: int) -> int:
    text = _raw_option(cfg, option)
    if text is None:
        return int(default)
    return int(text)


def _optional_int_option(cfg, option: str) -> Optional[int]:
    text = _raw_option(cfg, option)
    if text is None:
        return None
    return int(text)


def resolve_free_run_metric_settings(cfg=None) -> dict[str, Any]:
    """``[Evaluation]`` knobs. Missing keys keep the defaults below.

    ``rosenstein_embed_dim``, ``rosenstein_tau``, and ``rosenstein_theiler``
    stay unset unless the config names them. Unset means: embedding dimension
    and delay from the existing delay-embedding choice (fixed table, CLI, or
    the TwoNN / ID scan), and Theiler window ``(m - 1) * tau``.
    """
    return {
        "emd_max_samples": _int_option(cfg, "emd_max_samples", DEFAULT_EMD_MAX_SAMPLES),
        "emd_n_projections": _int_option(
            cfg, "emd_n_projections", DEFAULT_EMD_N_PROJECTIONS
        ),
        "emd_seed": _int_option(cfg, "emd_seed", DEFAULT_EMD_SEED),
        "emd_exact_max_n": _int_option(cfg, "emd_exact_max_n", DEFAULT_EMD_EXACT_MAX_N),
        "emd_exact_max_dim": _int_option(
            cfg, "emd_exact_max_dim", DEFAULT_EMD_EXACT_MAX_DIM
        ),
        "rosenstein_embed_dim": _optional_int_option(cfg, "rosenstein_embed_dim"),
        "rosenstein_tau": _optional_int_option(cfg, "rosenstein_tau"),
        "rosenstein_theiler": _optional_int_option(cfg, "rosenstein_theiler"),
        "rosenstein_evolution": _int_option(
            cfg, "rosenstein_evolution", DEFAULT_EVOLUTION
        ),
        "rosenstein_fit_steps": _optional_int_option(cfg, "rosenstein_fit_steps"),
        "rosenstein_fit_offset": _int_option(cfg, "rosenstein_fit_offset", 0),
    }


def _nan_distance() -> dict[str, Any]:
    nan = float("nan")
    return {
        "emd_w1_tf": nan,
        "emd_w1_auto": nan,
        "emd_sliced_tf": nan,
        "emd_sliced_auto": nan,
        "emd_exact_tf": nan,
        "emd_exact_auto": nan,
        "per_dim_tf": {},
        "per_dim_auto": {},
    }


def _nan_lle() -> dict[str, Any]:
    nan = float("nan")
    return {
        "per_channel": {},
        "lle_rosenstein_gt": nan,
        "lle_rosenstein_auto": nan,
        "lle_rosenstein_diff": nan,
    }


def free_run_window_metrics(
    channel_benchmarks: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Scalar scores for the YAML merge, plus bookkeeping that is not averaged.

    A failure in one family leaves that family NaN and records a skip reason.
    The other family still runs.
    """
    try:
        distances = distribution_distances_on_benchmarks(
            channel_benchmarks,
            max_samples=int(settings["emd_max_samples"]),
            n_projections=int(settings["emd_n_projections"]),
            seed=int(settings["emd_seed"]),
            exact_max_n=int(settings["emd_exact_max_n"]),
            exact_max_dim=int(settings["emd_exact_max_dim"]),
        )
        distance_error = None
    except Exception as exc:
        distances = _nan_distance()
        distance_error = str(exc)
    try:
        lyap = rosenstein_on_benchmarks(
            channel_benchmarks,
            embed_dim=settings.get("rosenstein_embed_dim"),
            tau=settings.get("rosenstein_tau"),
            theiler=settings.get("rosenstein_theiler"),
            evolution=int(settings["rosenstein_evolution"]),
            fit_steps=settings.get("rosenstein_fit_steps"),
            fit_offset=int(settings.get("rosenstein_fit_offset") or 0),
        )
        lyap_error = None
    except Exception as exc:
        lyap = _nan_lle()
        lyap_error = str(exc)

    scalars: dict[str, float] = {}
    scalars.update(flatten_distribution_distances(distances))
    scalars.update(flatten_rosenstein(lyap))

    book: dict[str, Any] = {
        "emd_max_samples": int(settings["emd_max_samples"]),
        "emd_n_projections": int(settings["emd_n_projections"]),
        "emd_exact_max_n": int(settings["emd_exact_max_n"]),
        "emd_exact_max_dim": int(settings["emd_exact_max_dim"]),
        "lle_rosenstein_evolution": int(settings["rosenstein_evolution"]),
    }
    for key in ("emd_exact_skip_reason_tf", "emd_exact_skip_reason_auto"):
        reason = distances.get(key)
        if reason:
            book[key] = reason
    if distance_error:
        book["emd_exact_skip_reason"] = distance_error
    if lyap_error:
        book["lle_rosenstein_skip_reason"] = lyap_error
    for key in (
        "lle_rosenstein_embed_dim",
        "lle_rosenstein_tau",
        "lle_rosenstein_theiler",
        "lle_rosenstein_dt",
    ):
        if key in lyap and lyap[key] is not None and np.isfinite(float(lyap[key])):
            book[key] = lyap[key]
    if settings.get("rosenstein_fit_steps") is not None:
        book["lle_rosenstein_fit_steps"] = int(settings["rosenstein_fit_steps"])
    return scalars, book
