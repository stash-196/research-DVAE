"""Aggregate per-batch evaluation metrics into YAML-ready summaries."""

from typing import Any, Dict, List

import numpy as np


def _nanmean_std(values: List[float]):
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def merge_batch_metric_dicts(batch_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a list of per-batch scalar dicts into mean/std summaries."""
    if not batch_metrics:
        return {}

    all_keys = set()
    for bm in batch_metrics:
        all_keys.update(bm.keys())

    merged: Dict[str, Any] = {"n_eval_batches": len(batch_metrics)}
    for key in sorted(all_keys):
        if key.startswith("_"):
            continue
        values = [bm[key] for bm in batch_metrics if key in bm]
        scalars = []
        for v in values:
            try:
                scalars.append(float(v))
            except (TypeError, ValueError):
                scalars = []
                break
        if scalars:
            mean, std = _nanmean_std(scalars)
            merged[key] = mean
            merged[f"{key}_std_across_batches"] = std

    return merged


def flatten_analysis_to_batch_metrics(
    mse_results: Dict[str, Any],
    geom_results: Dict[str, Any],
    spectrum_results: Dict[str, Any],
) -> Dict[str, float]:
    """Flatten one batch's analysis outputs into scalar metrics."""
    out: Dict[str, float] = {}

    for key in ("mse_tf", "mse_auto", "mse_tf_mean", "mse_auto_mean"):
        if key in mse_results and mse_results[key] is not None:
            out[key] = float(mse_results[key])

    for ch_key, vals in mse_results.get("per_channel", {}).items():
        if "mse_tf" in vals and np.isfinite(vals["mse_tf"]):
            out[f"mse_tf_{ch_key}"] = float(vals["mse_tf"])
        if "mse_auto" in vals and np.isfinite(vals["mse_auto"]):
            out[f"mse_auto_{ch_key}"] = float(vals["mse_auto"])

    for key in ("kld_tf", "kld_auto", "kld_tf_mean", "kld_auto_mean"):
        if key in geom_results and geom_results[key] is not None:
            out[key] = float(geom_results[key])

    for ch_key, vals in geom_results.get("per_channel", {}).items():
        if "kld_tf" in vals and np.isfinite(vals["kld_tf"]):
            out[f"kld_tf_{ch_key}"] = float(vals["kld_tf"])
        if "kld_auto" in vals and np.isfinite(vals["kld_auto"]):
            out[f"kld_auto_{ch_key}"] = float(vals["kld_auto"])

    for key in (
        "spectrum_error_tf",
        "spectrum_error_auto",
        "spectrum_error_gt",
        "spectrum_error_tf_mean",
        "spectrum_error_auto_mean",
    ):
        if key in spectrum_results and spectrum_results[key] is not None:
            out[key] = float(spectrum_results[key])

    for ch_key, vals in spectrum_results.get("per_channel", {}).items():
        if "spectrum_error_tf" in vals and np.isfinite(vals["spectrum_error_tf"]):
            out[f"spectrum_error_tf_{ch_key}"] = float(vals["spectrum_error_tf"])
        if "spectrum_error_auto" in vals and np.isfinite(vals["spectrum_error_auto"]):
            out[f"spectrum_error_auto_{ch_key}"] = float(vals["spectrum_error_auto"])

    return out