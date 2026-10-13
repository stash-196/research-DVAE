"""Persist and redraw the series behind the training figures.

The live training loop draws loss, validation-criteria, warm-up, patience,
and windowing figures from in-memory arrays. Those arrays are written into
the checkpoint ``loss_log`` and into ``loss_model.pckl`` at every
``save_frequency`` and again at the end. A resumed run reloads them.
``regenerate_training_figures`` redraws the same figures from that file
alone: no dataset and no model.

Series that were not persisted before this module (checkpoint or pickle):

* ``auto_warm_values``, ``noise_warm_values``, ``kl_warm_values``
  (the pickle stored only ``kl_warm_epochs``, the change indices)
* ``sequence_len_values``
* ``cpt_patience_epochs``
* ``patience_threshold_epochs`` (the active patience drawn as a dashed line)
* ``delta_per_epoch``
* ``best_state_epochs``
* the windowing step series ``windows_kept``, ``windows_dropped_nan``,
  ``windows_dropped_shift``, ``gap_splits``, ``mean_start_shift``
* the one-time dataset stats ``dataset_missing_rate``,
  ``nan_ratio_threshold``, ``gap_quantile_len``, ``gap_split_threshold``,
  ``block_mask_fraction``

``val_spectrum`` and ``val_criteria_score`` were stored when those criteria
were on, but a reload did not copy them back into the running arrays.
``sigmas_history`` / ``A_sigmas_history`` were in the final pickle only.

A new history always stores the raw ``val_spectrum`` series (not divided
by the first epoch) and ``val_metric_reference``, the first-epoch recon
and spectrum values. ``spectrum_monitored`` is false only when the legacy
switch skipped the spectrum windows. The validation figure is redrawn
from those fields: raw recon, raw spectrum error when monitoring was on,
and the stored selection score when more than one criterion was selected.

When those windows were built, the history also stores the Appendix B
terms ``val_base_error``, ``val_auto_drift``, ``val_error_drift_cross``,
and ``val_error_drift_total``, the same terms divided by the per-channel
variance of ``x_true`` (``*_varnorm``), ``val_auto_drift_k1``,
``val_auto_drift_k5``, and ``val_auto_drift_k``, ``val_recon_k`` (the
autonomous MSE on those same K entries), ``val_ftle``, and when
Rosenstein is on ``val_lle_mean`` / ``val_lle_std`` plus the scalar
``val_lle_gt``. ``val_ftle_per_window`` and ``val_lle_per_window`` keep
one value per fixed window; ``val_lle_gt_per_window`` is that same
ground-truth exponent per window, computed once. The plotted Rosenstein
band is ``val_lle_std``. Drift figures are written raw and as
``*_varnorm``. The per-timestep trajectories of ``e^2``, ``e d``, ``d^2``, and
the cosine of ``e`` and ``d`` are drawn at each ``save_frequency`` checkpoint into
``vis_during_training/drift_timestep/`` (epoch in the filename). Those
arrays are not stored here; they can be recomputed from the saved
model. The chosen window starts and their teacher-forced
observed fraction, longest missing run, and qualified flag are stored
once. ``drift_monitored`` is false when the windows were not built, and
the figure then omits those panels. Older files that still use
``val_drift``, ``val_cross``, ``val_auto_total``, and ``val_drift_k*``
are read as the new names.

Every new history also stores per-epoch wall times (``time_train``,
``time_val``, ``time_val_forward``, ``time_spectrum``, ``time_drift``,
``time_rosenstein``, ``time_plot``, ``time_eval``) and ``bump_events``,
the warm-up table ``(epoch, type, old, new, patience)`` the vertical
lines are rebuilt from. Subband and decorr rows in that table use the
same columns. Their ``old`` and ``new`` values are the warm-up scale.

When a training run enables subband or decorr, the history also stores
that term's warm-up scale (``subband_warm_values`` /
``decorr_warm_values``, the 0–1 progress) and its effective weight
(``subband_loss_weight_values`` / ``decorr_loss_weight_values``) beside
the autonomous, KL, noise, window, and patience series.
``subband_enabled`` and ``decorr_enabled`` say whether the warm-up
panel should draw them. A disabled term stays out of that panel and
off the bump lines.
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

from dvae.regularizers import OPTIONAL_HISTORY_KEYS

# Appendix B series. Stored when the fixed windows were scored.
DRIFT_SERIES_KEYS = (
    "val_base_error",
    "val_auto_drift",
    "val_error_drift_cross",
    "val_error_drift_total",
    "val_base_error_varnorm",
    "val_auto_drift_varnorm",
    "val_error_drift_cross_varnorm",
    "val_error_drift_total_varnorm",
    "val_auto_drift_k1",
    "val_auto_drift_k5",
    "val_auto_drift_k",
    "val_recon_k",
    "val_ftle",
    "val_lle_mean",
    "val_lle_std",
)

# Names written before the rename. A reload copies them onto the new keys
# when the new key is absent, so an older pickle still replots.
_DRIFT_KEY_ALIASES = {
    "val_drift": "val_auto_drift",
    "val_cross": "val_error_drift_cross",
    "val_auto_total": "val_error_drift_total",
    "val_drift_varnorm": "val_auto_drift_varnorm",
    "val_cross_varnorm": "val_error_drift_cross_varnorm",
    "val_auto_total_varnorm": "val_error_drift_total_varnorm",
    "val_drift_k1": "val_auto_drift_k1",
    "val_drift_k5": "val_auto_drift_k5",
    "val_drift_k": "val_auto_drift_k",
}

# Epoch series the training figures read. Present in every new history file.
EPOCH_SERIES_KEYS = (
    "train_loss",
    "val_loss",
    "train_recon",
    "train_kl",
    "val_recon",
    "val_kl",
    "val_spectrum",
    "val_criteria_score",
    "kl_warm_values",
    "auto_warm_values",
    "noise_warm_values",
    "sequence_len_values",
    "cpt_patience_epochs",
    "patience_threshold_epochs",
    "delta_per_epoch",
    "best_state_epochs",
    "windows_kept",
    "windows_dropped_nan",
    "windows_dropped_shift",
    "gap_splits",
    "mean_start_shift",
    "time_train",
    "time_val",
    "time_val_forward",
    "time_spectrum",
    "time_drift",
    "time_rosenstein",
    "time_plot",
    "time_eval",
)

# Held across the unfilled tail so a resumed array does not invent a step.
HOLD_FILL_KEYS = frozenset(
    {
        "kl_warm_values",
        "auto_warm_values",
        "noise_warm_values",
        "sequence_len_values",
        "windows_kept",
        "windows_dropped_nan",
        "windows_dropped_shift",
        "gap_splits",
        "mean_start_shift",
        "patience_threshold_epochs",
        "best_state_epochs",
        "subband_warm_values",
        "subband_loss_weight_values",
        "decorr_warm_values",
        "decorr_loss_weight_values",
    }
)

DATASET_STAT_KEYS = (
    "dataset_missing_rate",
    "nan_ratio_threshold",
    "nan_ratio_threshold_state",
    "gap_quantile_len",
    "gap_split_threshold",
    "gap_split_threshold_state",
    "block_mask_fraction",
)

# What a history file written before this module left out.
NOT_PERSISTED_BEFORE = (
    "auto_warm_values",
    "noise_warm_values",
    "kl_warm_values",
    "sequence_len_values",
    "cpt_patience_epochs",
    "patience_threshold_epochs",
    "delta_per_epoch",
    "best_state_epochs",
    "windows_kept",
    "windows_dropped_nan",
    "windows_dropped_shift",
    "gap_splits",
    "mean_start_shift",
    "dataset_missing_rate",
    "nan_ratio_threshold",
    "gap_quantile_len",
    "gap_split_threshold",
    "block_mask_fraction",
)


def warmup_change_epochs(values) -> np.ndarray:
    """Epochs where a step series changes, with a leading 0.

    Computed on the saved prefix, so the unfilled tail of a full-length
    buffer does not add a marker.
    """
    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size == 0:
        return np.asarray([0], dtype=int)
    changes = np.where(np.diff(series) != 0)[0] + 1
    return np.insert(changes, 0, 0)


def _slice_1d(values, end):
    arr = np.asarray(values)
    if end is None:
        return arr
    return arr[: int(end)]


def _slice_epoch_rows(values, end):
    """Keep epoch rows of an ``(epoch, window)`` array."""
    arr = np.asarray(values)
    if end is None:
        return arr
    return arr[: int(end)]


def _slice_2d_epochs(values, end):
    if values is None:
        return None
    arr = np.asarray(values)
    if end is None:
        return arr
    return arr[:, : int(end)]


def _plain_stats(dataset_stats) -> dict:
    raw = {} if dataset_stats is None else dict(dataset_stats)
    out = {}
    for key in DATASET_STAT_KEYS:
        value = raw.get(key)
        if value is None or isinstance(value, str):
            out[key] = value
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            out[key] = None
            continue
        out[key] = number if np.isfinite(number) else None
    return out


def _float_reference(reference):
    """Copy a criterion map to plain floats. Empty stays unset."""
    if not reference:
        return None
    out = {}
    for name, value in dict(reference).items():
        out[str(name)] = float(value)
    return out or None


def build_training_history(
    *,
    end,
    model_name,
    tag,
    auto_sampling_method,
    noise_sampling_method,
    delta_threshold,
    val_criteria,
    val_reference,
    series,
    dataset_stats,
    sigmas_history=None,
    a_sigmas_history=None,
    metric_reference=None,
    spectrum_monitored=True,
    drift_monitored=False,
    drift_horizon=None,
    drift_flip=None,
    x_true_var=None,
    window_info=None,
    lle_gt=None,
    rosenstein=None,
    ftle_fit=None,
    lle_per_window=None,
    ftle_per_window=None,
    lle_gt_per_window=None,
    bump_events=None,
    subband_enabled=None,
    decorr_enabled=None,
) -> dict:
    """One dict for the checkpoint ``loss_log`` and for ``loss_model.pckl``.

    ``end`` is the exclusive epoch bound (the epoch just finished, plus one).
    ``None`` keeps each array as given. ``val_loss`` and ``val_spectrum``
    stay the raw per-epoch values. ``val_metric_reference`` is the
    first-epoch normaliser (recon, and spectrum when it was scored).
    ``metric_reference`` wins; otherwise a non-empty ``val_reference`` is
    stored under that same key. ``spectrum_monitored`` records whether the
    spectrum windows were built.
    """
    from dvae.run_options import criteria_log_fields
    from dvae.timescale_history import attach_timescale_history

    history = {}
    for key in EPOCH_SERIES_KEYS:
        history[key] = _slice_1d(series[key], end)
    names = tuple(val_criteria)
    history["val_criteria"] = ",".join(names)
    history["model_name"] = "" if model_name is None else str(model_name)
    history["tag"] = "" if tag is None else str(tag)
    history["auto_sampling_method"] = (
        "none" if auto_sampling_method is None else str(auto_sampling_method)
    )
    history["noise_sampling_method"] = (
        "none" if noise_sampling_method is None else str(noise_sampling_method)
    )
    history["delta_threshold"] = float(delta_threshold)
    history["spectrum_monitored"] = bool(spectrum_monitored)
    history["drift_monitored"] = bool(drift_monitored)
    for key in DRIFT_SERIES_KEYS:
        if key in series:
            history[key] = _slice_1d(series[key], end)
    # Subband and decorr are optional. Callers that never pass them
    # (older tests, older checkpoints) keep working. A training run
    # passes the scale and the effective weight every epoch, and sets
    # the enabled flags so a redraw can hide a term that was off.
    # Count of free-run spectrum segments skipped for too few finite samples.
    if "val_spectrum_skipped" in series:
        history["val_spectrum_skipped"] = _slice_1d(
            series["val_spectrum_skipped"], end
        )
    for key in OPTIONAL_HISTORY_KEYS:
        if key in series:
            history[key] = _slice_1d(series[key], end)
    if subband_enabled is not None:
        history["subband_enabled"] = bool(subband_enabled)
    if decorr_enabled is not None:
        history["decorr_enabled"] = bool(decorr_enabled)
    if "subband_warm_values" in history:
        history["subband_warm_epochs"] = warmup_change_epochs(
            history["subband_warm_values"]
        )
    if "decorr_warm_values" in history:
        history["decorr_warm_epochs"] = warmup_change_epochs(
            history["decorr_warm_values"]
        )
    if drift_horizon is not None:
        history["val_drift_horizon"] = int(drift_horizon)
    if drift_flip is not None:
        history["val_drift_flip"] = int(drift_flip)
    if x_true_var is not None:
        history["val_x_true_var"] = np.asarray(x_true_var, dtype=float).reshape(-1)
    if window_info:
        history["val_window_starts"] = np.asarray(window_info["starts"], dtype=np.int64)
        history["val_window_tf_observed"] = np.asarray(
            window_info["tf_observed"], dtype=float
        )
        history["val_window_tf_max_gap"] = np.asarray(
            window_info["tf_max_gap"], dtype=np.int64
        )
        history["val_window_qualified"] = np.asarray(
            window_info["qualified"], dtype=bool
        )
        history["val_window_relaxed"] = bool(window_info["relaxed"])
        history["val_window_tf_min_observed"] = float(window_info["min_observed"])
        history["val_window_tf_max_gap_frac"] = float(window_info["max_gap_frac"])
        history["val_window_seed"] = int(window_info["seed"])
    if lle_gt is not None and np.isfinite(float(lle_gt)):
        history["val_lle_gt"] = float(lle_gt)
    if lle_per_window is not None:
        history["val_lle_per_window"] = _slice_epoch_rows(lle_per_window, end)
    if ftle_per_window is not None:
        history["val_ftle_per_window"] = _slice_epoch_rows(ftle_per_window, end)
    if lle_gt_per_window is not None:
        history["val_lle_gt_per_window"] = np.asarray(
            lle_gt_per_window, dtype=float
        ).reshape(-1)
    history["bump_events"] = _plain_bump_events(bump_events, end)
    if rosenstein is not None:
        history["val_rosenstein"] = bool(rosenstein)
    if ftle_fit is not None:
        history["val_ftle_fit_start"] = int(ftle_fit[0])
        history["val_ftle_fit_end"] = int(ftle_fit[1])
    metric = _float_reference(metric_reference)
    if metric is None:
        metric = _float_reference(val_reference)
    if metric:
        history["val_metric_reference"] = metric
    history["dataset_stats"] = _plain_stats(dataset_stats)
    shown = history["kl_warm_values"]
    history["kl_warm_epochs"] = warmup_change_epochs(shown)
    history["auto_warm_epochs"] = warmup_change_epochs(history["auto_warm_values"])
    history["noise_warm_epochs"] = warmup_change_epochs(history["noise_warm_values"])
    history["sequence_len_epochs"] = warmup_change_epochs(history["sequence_len_values"])
    stop = shown.shape[0]
    history.update(
        criteria_log_fields(
            names,
            history["val_spectrum"],
            history["val_criteria_score"],
            val_reference,
            stop,
        )
    )
    # criteria_log_fields omits val_criteria for recon-only. Keep the key.
    history["val_criteria"] = ",".join(names)
    attach_timescale_history(
        history,
        sigmas_history=_slice_2d_epochs(sigmas_history, end),
        a_sigmas_history=_slice_2d_epochs(a_sigmas_history, end),
    )
    return history


def _plain_bump_events(events, end) -> list:
    """Copy bump rows with epoch < ``end``.

    Each row is ``epoch``, ``type`` (window, auto, noise, or KL),
    ``old``, ``new``, and ``patience`` (the active patience that fired
    the bump). ``patience_after`` and ``cpt_patience`` keep the rest of
    the log line. A missing table is an empty list.
    """
    rows = []
    if not events:
        return rows
    stop = None if end is None else int(end)
    for item in events:
        epoch = int(item["epoch"])
        if stop is not None and epoch >= stop:
            continue
        rows.append(
            {
                "epoch": epoch,
                "type": str(item["type"]),
                "old": float(item["old"]),
                "new": float(item["new"]),
                "patience": float(item["patience"]),
                "patience_after": float(item.get("patience_after", item["patience"])),
                "cpt_patience": int(item.get("cpt_patience", 0)),
            }
        )
    return rows


def restore_epoch_series(saved, epochs, fill="zero") -> np.ndarray:
    """Length-``epochs`` copy of a saved 1-d series.

    ``fill='hold'`` repeats the last saved value over the unfilled tail.
    ``fill='zero'`` pads with zeros, matching the historical loss arrays.
    """
    values = np.asarray(saved, dtype=np.float64).reshape(-1)
    epochs = int(epochs)
    if values.size == 0 or epochs <= 0:
        return np.zeros(max(epochs, 0), dtype=np.float64)
    pad = float(values[-1]) if fill == "hold" else 0.0
    out = np.full(epochs, pad, dtype=np.float64)
    n = min(values.size, epochs)
    out[:n] = values[:n]
    return out


def restore_matrix(saved, shape) -> np.ndarray:
    """Copy a saved ``(units, epochs)`` trace into a fresh buffer of ``shape``."""
    rows, cols = int(shape[0]), int(shape[1])
    out = np.zeros((rows, cols), dtype=np.float64)
    arr = np.asarray(saved, dtype=np.float64)
    if arr.ndim != 2 or rows == 0 or cols == 0:
        return out
    r = min(rows, arr.shape[0])
    c = min(cols, arr.shape[1])
    out[:r, :c] = arr[:r, :c]
    return out


def restore_per_window(saved, epochs, n_windows) -> np.ndarray:
    """Copy an ``(epoch, window)`` Lyapunov trace into a length-``epochs`` buffer.

    The window count is fixed at training start. A different width raises.
    Missing epochs stay NaN.
    """
    epochs = int(epochs)
    n_windows = int(n_windows)
    out = np.full((epochs, n_windows), np.nan, dtype=np.float64)
    arr = np.asarray(saved, dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[1]) != n_windows:
        width = None if arr.ndim != 2 else int(arr.shape[1])
        raise ValueError(
            "per-window Lyapunov was saved with "
            f"{width} windows; this resume has {n_windows}."
        )
    rows = min(epochs, arr.shape[0])
    out[:rows] = arr[:rows]
    return out


def series_or_new(loss_log, key, epochs, fill="zero") -> np.ndarray:
    """Restored series, or a fresh zero buffer when the checkpoint lacks it."""
    if isinstance(loss_log, dict) and key in loss_log:
        chosen = fill
        if fill == "auto":
            chosen = "hold" if key in HOLD_FILL_KEYS else "zero"
        return restore_epoch_series(loss_log[key], epochs, fill=chosen)
    return np.zeros(int(epochs), dtype=np.float64)


def save_training_history(path, history) -> None:
    """Write the history dict as a pickle (``loss_model.pckl``)."""
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(history, handle)


def load_training_history(path) -> dict:
    """Load ``loss_model.pckl`` or a ``.pt`` checkpoint's ``loss_log``.

    A checkpoint is used only for its history dict. The model weights are
    not required and are not returned.
    """
    path = os.fspath(path)
    if path.endswith((".pt", ".pth")):
        import torch

        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        if isinstance(blob, dict) and isinstance(blob.get("loss_log"), dict):
            return blob["loss_log"]
        raise ValueError(
            f"{path} has no loss_log dict. Pass loss_model.pckl or a "
            "training checkpoint."
        )
    with open(path, "rb") as handle:
        loaded = pickle.load(handle)
    if isinstance(loaded, dict) and "train_loss" not in loaded and isinstance(
        loaded.get("loss_log"), dict
    ):
        return loaded["loss_log"]
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not a training-history dict.")
    return loaded


def _criteria_names(history):
    from dvae.run_options import canonical_criterion_name

    raw = history.get("val_criteria")
    if raw is None or str(raw).strip() == "":
        return ("recon",)
    names = []
    for part in str(raw).split(","):
        canon = canonical_criterion_name(part)
        if canon and canon not in names:
            names.append(canon)
    return tuple(names) or ("recon",)


def drift_series_source(loss_log):
    """History dict with pre-rename drift series copied onto the current keys."""
    if not isinstance(loss_log, dict):
        return loss_log
    return _with_drift_aliases(loss_log)


def _with_drift_aliases(history) -> dict:
    """Copy pre-rename drift keys onto the current names when those are absent."""
    out = dict(history)
    for old, new in _DRIFT_KEY_ALIASES.items():
        if old in out and new not in out:
            out[new] = out[old]
    reference = out.get("val_metric_reference")
    if isinstance(reference, dict):
        from dvae.run_options import canonical_criterion_name

        mapped = {}
        for name, value in reference.items():
            key = canonical_criterion_name(name)
            if key not in mapped or str(name) == key:
                mapped[key] = value
        out["val_metric_reference"] = mapped
    return out


def _best_epoch_index(history, n):
    """Last saved best-epoch index, when it lands inside the plotted prefix."""
    saved = history.get("best_state_epochs")
    if saved is None:
        return None
    arr = np.asarray(saved, dtype=float).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr[-1]):
        return None
    idx = int(arr[-1])
    if abs(float(arr[-1]) - idx) > 1e-6:
        return None
    if 0 <= idx < int(n):
        return idx
    return None


def _drift_for_figure(history, n):
    """Raw and variance-normalised Appendix B series, or None when absent.

    A file with ``drift_monitored`` false does not grow zero panels.
    Older pickles without the series stay on recon and spectrum only.
    """
    history = _with_drift_aliases(history)
    if "drift_monitored" in history and not bool(history["drift_monitored"]):
        return None
    if "val_base_error" not in history and "val_auto_drift" not in history:
        return None
    from dvae.eval.utils.local_drift_analysis import (
        DRIFT_PLOT_NAMES,
        VARNORM_DRIFT_METRICS,
    )

    out = {}
    for name in DRIFT_PLOT_NAMES + VARNORM_DRIFT_METRICS:
        key = "val_" + name
        if key not in history:
            continue
        out[name] = np.asarray(history[key], dtype=float).reshape(-1)[: int(n)]
    return out or None


def _recon_k_for_figure(history, n):
    history = _with_drift_aliases(history)
    if "drift_monitored" in history and not bool(history["drift_monitored"]):
        return None
    if "val_recon_k" not in history:
        return None
    return np.asarray(history["val_recon_k"], dtype=float).reshape(-1)[: int(n)]


def _lyapunov_for_figure(history, n):
    history = _with_drift_aliases(history)
    if "drift_monitored" in history and not bool(history["drift_monitored"]):
        return None
    if "val_ftle" not in history and "val_lle_mean" not in history:
        return None

    def _opt(key):
        if key not in history:
            return None
        return np.asarray(history[key], dtype=float).reshape(-1)[: int(n)]

    gt = history.get("val_lle_gt")
    try:
        gt_value = None if gt is None else float(gt)
    except (TypeError, ValueError):
        gt_value = None
    return {
        "ftle": _opt("val_ftle"),
        "lle_mean": _opt("val_lle_mean"),
        "lle_std": _opt("val_lle_std"),
        "lle_gt": gt_value,
    }


def _timing_for_figure(history, n):
    """Per-epoch wall times, or None when this file has none of them.

    An older pickle keeps the seven-panel training figure. A new file
    draws the eighth panel from whatever of these keys it stored.
    """
    from dvae.epoch_log import TIMING_SERIES_KEYS

    if not any(key in history for key in TIMING_SERIES_KEYS):
        return None
    out = {}
    for key in TIMING_SERIES_KEYS:
        if key not in history:
            continue
        out[key] = np.asarray(history[key], dtype=float).reshape(-1)[: int(n)]
    return out or None


def _spectrum_for_figure(history, criteria):
    """Whether the redraw includes spectrum, and the raw series or None.

    A file that records ``spectrum_monitored`` uses that flag. Older
    pickles include spectrum only when the series or the criterion is
    present. A missing series is left as None so the figure does not
    invent a zero line.
    """
    if "spectrum_monitored" in history:
        include = bool(history["spectrum_monitored"])
    else:
        include = "val_spectrum" in history or "spectrum" in criteria
    if include and "val_spectrum" in history:
        return True, np.asarray(history["val_spectrum"], dtype=float).reshape(-1)
    return False, None


def regenerate_training_figures(history, save_figures_dir, tag=None, drift_norm="both"):
    """Redraw the training figures from a history dict. No data and no model.

    Parameter heatmaps, the long-sequence batch figure, and the
    per-timestep drift trajectories need a model, so they are not part
    of this redraw. Everything else the
    training loop saves at ``save_frequency`` is: total loss, validation
    criteria, reconstruction and KL when the model is a VRNN, the combined
    warm-up / patience / windowing figure, and sigma / alpha history when
    those traces are in the file.
    """
    from dvae.timescale_history import save_plrnn_timescale_figures
    from dvae.visualizers.visualize_training_metrics import (
        visualize_alpha_history,
        visualize_combined_metrics,
        visualize_kld_loss,
        visualize_recon_loss,
        visualize_sigma_history,
        visualize_total_loss,
        visualize_val_criteria,
    )

    if not isinstance(history, dict):
        raise ValueError("training history must be a dict.")
    os.makedirs(save_figures_dir, exist_ok=True)
    before = set(os.listdir(save_figures_dir))
    file_tag = tag if tag is not None else history.get("tag") or "train"
    model_name = str(history.get("model_name") or "")
    auto_method = str(history.get("auto_sampling_method") or "none")
    noise_method = str(history.get("noise_sampling_method") or "none")
    if "train_loss" not in history or "val_loss" not in history:
        raise ValueError("training history is missing train_loss or val_loss.")
    train_loss = np.asarray(history["train_loss"], dtype=float).reshape(-1)
    val_loss = np.asarray(history["val_loss"], dtype=float).reshape(-1)
    n = int(min(train_loss.size, val_loss.size))
    train_loss = train_loss[:n]
    val_loss = val_loss[:n]

    def _series(key):
        if key not in history:
            return np.zeros(n, dtype=float)
        return np.asarray(history[key], dtype=float).reshape(-1)[:n]

    def _markers(key, values_key):
        if key in history:
            return np.asarray(history[key]).reshape(-1)
        return warmup_change_epochs(_series(values_key))

    kl_warm_epochs = _markers("kl_warm_epochs", "kl_warm_values")
    auto_warm_epochs = _markers("auto_warm_epochs", "auto_warm_values")
    noise_warm_epochs = _markers("noise_warm_epochs", "noise_warm_values")
    sequence_len_epochs = _markers("sequence_len_epochs", "sequence_len_values")

    def _enabled_regularizer(prefix):
        """Scale, effective weight, and bump epochs, or Nones when hidden."""
        if not history.get(prefix + "_enabled"):
            return None, None, None
        scale_key = prefix + "_warm_values"
        weight_key = prefix + "_loss_weight_values"
        if scale_key not in history and weight_key not in history:
            return None, None, None
        scale = _series(scale_key) if scale_key in history else None
        weight = _series(weight_key) if weight_key in history else None
        if prefix + "_warm_epochs" in history:
            epochs = np.asarray(history[prefix + "_warm_epochs"]).reshape(-1)
        elif scale_key in history:
            epochs = warmup_change_epochs(scale)
        else:
            epochs = warmup_change_epochs(weight)
        return scale, weight, epochs

    sub_scale, sub_weight, sub_epochs = _enabled_regularizer("subband")
    dec_scale, dec_weight, dec_epochs = _enabled_regularizer("decorr")
    delta_threshold = float(history.get("delta_threshold", 1e-4))
    criteria = _criteria_names(history)
    dataset_stats = history.get("dataset_stats")
    if not isinstance(dataset_stats, dict):
        dataset_stats = None

    visualize_total_loss(
        train_loss,
        val_loss,
        kl_warm_epochs,
        auto_warm_epochs,
        noise_warm_epochs,
        sequence_len_epochs,
        model_name,
        auto_method,
        noise_method,
        save_figures_dir,
        file_tag,
        subband_warm_epochs=sub_epochs,
        decorr_warm_epochs=dec_epochs,
    )
    visualize_total_loss(
        train_loss,
        val_loss,
        kl_warm_epochs,
        auto_warm_epochs,
        noise_warm_epochs,
        sequence_len_epochs,
        model_name,
        auto_method,
        noise_method,
        save_figures_dir,
        file_tag,
        log_scale=True,
        subband_warm_epochs=sub_epochs,
        decorr_warm_epochs=dec_epochs,
    )
    include_spectrum, spectrum = _spectrum_for_figure(history, criteria)
    if spectrum is not None:
        spectrum = spectrum[:n]
    score = (
        _series("val_criteria_score")
        if "val_criteria_score" in history
        else None
    )
    visualize_val_criteria(
        criteria,
        val_loss,
        spectrum,
        score,
        kl_warm_epochs,
        auto_warm_epochs,
        noise_warm_epochs,
        sequence_len_epochs,
        model_name,
        auto_method,
        noise_method,
        save_figures_dir,
        file_tag,
        best_epoch=_best_epoch_index(history, n),
        include_spectrum=include_spectrum,
        drift=_drift_for_figure(history, n),
        recon_k=_recon_k_for_figure(history, n),
        lyapunov=_lyapunov_for_figure(history, n),
        drift_norm=drift_norm,
        subband_warm_epochs=sub_epochs,
        decorr_warm_epochs=dec_epochs,
    )
    if model_name in ("VRNN", "MT_VRNN"):
        visualize_recon_loss(
            _series("train_recon"),
            _series("val_recon"),
            kl_warm_epochs,
            auto_warm_epochs,
            noise_warm_epochs,
            sequence_len_epochs,
            model_name,
            auto_method,
            noise_method,
            save_figures_dir,
            file_tag,
            subband_warm_epochs=sub_epochs,
            decorr_warm_epochs=dec_epochs,
        )
        visualize_recon_loss(
            _series("train_recon"),
            _series("val_recon"),
            kl_warm_epochs,
            auto_warm_epochs,
            noise_warm_epochs,
            sequence_len_epochs,
            model_name,
            auto_method,
            noise_method,
            save_figures_dir,
            file_tag,
            log_scale=True,
            subband_warm_epochs=sub_epochs,
            decorr_warm_epochs=dec_epochs,
        )
        visualize_kld_loss(
            _series("train_kl"),
            _series("val_kl"),
            kl_warm_epochs,
            auto_warm_epochs,
            noise_warm_epochs,
            sequence_len_epochs,
            model_name,
            auto_method,
            noise_method,
            save_figures_dir,
            file_tag,
            subband_warm_epochs=sub_epochs,
            decorr_warm_epochs=dec_epochs,
        )
        visualize_kld_loss(
            _series("train_kl"),
            _series("val_kl"),
            kl_warm_epochs,
            auto_warm_epochs,
            noise_warm_epochs,
            sequence_len_epochs,
            model_name,
            auto_method,
            noise_method,
            save_figures_dir,
            file_tag,
            log_scale=True,
            subband_warm_epochs=sub_epochs,
            decorr_warm_epochs=dec_epochs,
        )
    patience = history.get("patience_threshold_epochs")
    visualize_combined_metrics(
        _series("delta_per_epoch"),
        delta_threshold,
        kl_warm_epochs,
        _series("kl_warm_values"),
        auto_warm_epochs,
        _series("auto_warm_values"),
        noise_warm_epochs,
        _series("noise_warm_values"),
        sequence_len_epochs,
        _series("sequence_len_values"),
        _series("cpt_patience_epochs"),
        _series("best_state_epochs"),
        model_name,
        auto_method,
        noise_method,
        save_figures_dir,
        file_tag,
        None if patience is None else _series("patience_threshold_epochs"),
        windows_kept=_series("windows_kept"),
        windows_dropped_nan=_series("windows_dropped_nan"),
        windows_dropped_shift=_series("windows_dropped_shift"),
        gap_splits=_series("gap_splits"),
        mean_start_shift=_series("mean_start_shift"),
        dataset_stats=dataset_stats,
        timing=_timing_for_figure(history, n),
        subband_warm_values=sub_scale,
        subband_loss_weight_values=sub_weight,
        decorr_warm_values=dec_scale,
        decorr_loss_weight_values=dec_weight,
        subband_warm_epochs=sub_epochs,
        decorr_warm_epochs=dec_epochs,
    )
    sigmas = history.get("sigmas_history")
    a_sigmas = history.get("A_sigmas_history")
    if sigmas is not None and (
        model_name in ("MT_RNN", "MT_VRNN") or a_sigmas is None
    ):
        sigmas = np.asarray(sigmas)
        if sigmas.ndim == 2 and sigmas.shape[1] >= n:
            sigmas = sigmas[:, :n]
        visualize_sigma_history(
            sigmas,
            model_name,
            save_figures_dir,
            file_tag,
            kl_warm_epochs,
            auto_warm_epochs if auto_method in ("ss", "sm") else None,
            noise_warm_epochs if noise_method in ("ss", "sm") else None,
            sequence_len_epochs,
            subband_warm_epochs=sub_epochs,
            decorr_warm_epochs=dec_epochs,
        )
        visualize_alpha_history(
            sigmas,
            model_name,
            save_figures_dir,
            file_tag,
            kl_warm_epochs,
            auto_warm_epochs if auto_method in ("ss", "sm") else None,
            noise_warm_epochs if noise_method in ("ss", "sm") else None,
            sequence_len_epochs,
            subband_warm_epochs=sub_epochs,
            decorr_warm_epochs=dec_epochs,
        )
    if a_sigmas is not None:
        a_sigmas = np.asarray(a_sigmas)
        if a_sigmas.ndim == 2 and a_sigmas.shape[1] >= n:
            a_sigmas = a_sigmas[:, :n]
        save_plrnn_timescale_figures(
            a_sigmas,
            model_name,
            save_figures_dir,
            file_tag,
            kl_warm_epochs,
            auto_warm_epochs if auto_method in ("ss", "sm") else None,
            noise_warm_epochs if noise_method in ("ss", "sm") else None,
            sequence_len_epochs,
            visualize_sigma_history=visualize_sigma_history,
            visualize_alpha_history=visualize_alpha_history,
            subband_warm_epochs=sub_epochs,
            decorr_warm_epochs=dec_epochs,
        )
    after = set(os.listdir(save_figures_dir))
    return [
        os.path.join(save_figures_dir, name) for name in sorted(after - before)
    ]


def main(argv=None) -> int:
    """CLI: redraw training figures from a saved history file."""
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate training figures from loss_model.pckl or from a "
            "checkpoint's loss_log. Does not load a dataset or a model."
        )
    )
    parser.add_argument(
        "history",
        help="Path to loss_model.pckl, or a .pt checkpoint that contains loss_log.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory for the regenerated png files.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Filename tag. Blank uses the tag stored in the history.",
    )
    parser.add_argument(
        "--drift-norm",
        choices=("raw", "varnorm", "both"),
        default="both",
        help=(
            "Which drift-term figures to write. both saves raw and the "
            "variance-normalised copies. Default: both."
        ),
    )
    args = parser.parse_args(argv)
    history = load_training_history(args.history)
    written = regenerate_training_figures(
        history, args.out_dir, tag=args.tag, drift_norm=args.drift_norm
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
