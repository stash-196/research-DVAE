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
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

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
) -> dict:
    """One dict for the checkpoint ``loss_log`` and for ``loss_model.pckl``.

    ``end`` is the exclusive epoch bound (the epoch just finished, plus one).
    ``None`` keeps each array as given.
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
    raw = history.get("val_criteria")
    if raw is None or str(raw).strip() == "":
        return ("recon",)
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def regenerate_training_figures(history, save_figures_dir, tag=None):
    """Redraw the training figures from a history dict. No data and no model.

    Parameter heatmaps and the long-sequence batch figure need a model and
    a batch, so they are not part of this redraw. Everything else the
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
    )
    visualize_val_criteria(
        criteria,
        val_loss,
        _series("val_spectrum"),
        _series("val_criteria_score"),
        kl_warm_epochs,
        auto_warm_epochs,
        noise_warm_epochs,
        sequence_len_epochs,
        model_name,
        auto_method,
        noise_method,
        save_figures_dir,
        file_tag,
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
    args = parser.parse_args(argv)
    history = load_training_history(args.history)
    written = regenerate_training_figures(history, args.out_dir, tag=args.tag)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
