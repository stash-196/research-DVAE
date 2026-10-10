"""Fixed validation windows scored by free-run spectral error.

Windows are copied from ``dataset.seq`` once. Nothing here calls
``update_sequence_length``, so the curriculum can still resize the live
loaders. The score is the existing Hellinger spectrum error on Hann-averaged
periodograms of the free-run half only.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from dvae.dataset.xhro_proper_dataset import (
    COARSE_WINDOW_RNG_SEED,
    intra_subject_window_starts,
)
from dvae.eval.utils.durstewitz_eval_metrics import power_spectrum_error
from dvae.eval.utils.forward_modes import run_forward_with_mode

_SPLIT_SEED = 1234567890


def _seq_array(dataset) -> np.ndarray:
    seq = dataset.seq
    if torch.is_tensor(seq):
        seq = seq.detach().cpu().numpy()
    arr = np.asarray(seq)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            "Spectrum validation needs a (time, channel) series, "
            f"got shape {arr.shape}."
        )
    return arr


def _group_index(dataset):
    group = getattr(dataset, "subject_index_per_row", None)
    if group is None:
        group = getattr(dataset, "stay_id_per_row", None)
    if group is None:
        return None
    return np.asarray(group)


def _candidate_starts(dataset, window_len: int) -> np.ndarray:
    """Starts whose ``window_len`` steps stay inside one subject or stay."""
    n = len(_seq_array(dataset))
    if window_len < 1 or n < window_len:
        return np.asarray([], dtype=np.int64)
    group = _group_index(dataset)
    if group is not None:
        return intra_subject_window_starts(
            group, int(window_len), overlap=bool(getattr(dataset, "overlap", False))
        )
    step = 1 if getattr(dataset, "overlap", False) else int(window_len)
    return np.arange(0, n - int(window_len) + 1, step, dtype=np.int64)


def _validation_starts(dataset, starts: np.ndarray) -> np.ndarray:
    """Keep the validation portion without touching the live loader.

    PhysioNet already put only validation stays on this dataset. Coarse and
    the waveform datasets still share the non-test series and split starts.
    """
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        return starts
    if getattr(dataset, "stay_id_per_row", None) is not None and not getattr(
        dataset, "coarse_mode", False
    ):
        return starts
    split = getattr(dataset, "split", "valid")
    if split == "test":
        return starts
    chosen = np.array(starts, copy=True)
    if getattr(dataset, "shuffle", False) and chosen.size:
        seed = (
            COARSE_WINDOW_RNG_SEED
            if getattr(dataset, "coarse_mode", False)
            else _SPLIT_SEED
        )
        rng = np.random.RandomState(seed)
        rng.shuffle(chosen)
    val_ratio = float(getattr(dataset, "val_indices", 0.0) or 0.0)
    split_point = int(len(chosen) * (1.0 - val_ratio))
    if split == "train":
        return chosen[:split_point]
    return chosen[split_point:]


def select_fixed_starts(starts: np.ndarray, n_windows: int) -> np.ndarray:
    """Evenly spaced, sorted starts. Fewer than ``n_windows`` returns all."""
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0 or n_windows < 1:
        return np.asarray([], dtype=np.int64)
    ordered = np.sort(starts)
    if ordered.size <= n_windows:
        return ordered
    pick = np.linspace(0, ordered.size - 1, int(n_windows)).round().astype(int)
    return ordered[np.unique(pick)]


def capture_fixed_spectrum_windows(
    dataset,
    n_windows: int = 10,
    window_len: int = 2000,
) -> list[np.ndarray]:
    """Copy fixed windows from the validation series.

    Raises when none fit. Uses every available start, and warns, when fewer
    than ``n_windows`` fit. Does not call ``update_sequence_length``.
    """
    window_len = int(window_len)
    n_windows = int(n_windows)
    if window_len < 2:
        raise ValueError(f"val_spectrum_window_len must be >= 2, got {window_len}.")
    if n_windows < 1:
        raise ValueError(f"val_spectrum_n_windows must be >= 1, got {n_windows}.")
    starts = _validation_starts(dataset, _candidate_starts(dataset, window_len))
    if starts.size == 0:
        raise ValueError(
            "val_criterion=spectrum needs at least one validation window of "
            f"length {window_len}; none fit inside a subject or stay."
        )
    chosen = select_fixed_starts(starts, n_windows)
    if chosen.size < n_windows:
        warnings.warn(
            f"Spectrum validation asked for {n_windows} windows of length "
            f"{window_len}; using all {chosen.size} that fit.",
            UserWarning,
            stacklevel=2,
        )
    series = _seq_array(dataset)
    return [np.array(series[int(s) : int(s) + window_len], copy=True) for s in chosen]


def _hann_periodogram(segment: np.ndarray) -> np.ndarray:
    x = np.asarray(segment, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    if not finite.any():
        raise ValueError("Spectrum segment has no finite samples.")
    center = float(np.mean(x[finite]))
    tapered = np.where(finite, x - center, 0.0)
    tapered = tapered * np.hanning(tapered.size)
    power = np.abs(np.fft.rfft(tapered)) ** 2
    if power.size < 2:
        raise ValueError("Spectrum segment is too short for a periodogram.")
    return power[1:]


def averaged_hann_spectrum(segments: list[np.ndarray]) -> np.ndarray:
    """Welch-style mean of Hann periodograms. Segments share a length."""
    spectra = [_hann_periodogram(segment) for segment in segments]
    return np.mean(np.stack(spectra, axis=0), axis=0)


def free_run_spectrum_error(
    predicted: list[np.ndarray],
    ground_truth: list[np.ndarray],
    tf_steps: int,
) -> float:
    """Hellinger spectrum error on the free-run half, mean over channels."""
    if len(predicted) != len(ground_truth) or not predicted:
        raise ValueError("Predicted and ground-truth window lists must match.")
    tf_steps = int(tf_steps)
    gt0 = np.asarray(ground_truth[0])
    if gt0.ndim == 1:
        gt0 = gt0.reshape(-1, 1)
    if tf_steps < 0 or tf_steps >= gt0.shape[0]:
        raise ValueError(
            f"val_spectrum_tf_steps must be in [0, window_len), got {tf_steps} "
            f"for window length {gt0.shape[0]}."
        )
    n_channels = int(gt0.shape[1])
    errors = []
    for channel in range(n_channels):
        gt_tail = []
        pred_tail = []
        for gt, pred in zip(ground_truth, predicted):
            gt_arr = np.asarray(gt)
            pred_arr = np.asarray(pred)
            if gt_arr.ndim == 1:
                gt_arr = gt_arr.reshape(-1, 1)
            if pred_arr.ndim == 1:
                pred_arr = pred_arr.reshape(-1, 1)
            gt_tail.append(gt_arr[tf_steps:, channel])
            pred_tail.append(pred_arr[tf_steps:, channel])
        gt_ps = averaged_hann_spectrum(gt_tail)
        pred_ps = averaged_hann_spectrum(pred_tail)
        err = power_spectrum_error(
            pred_ps.reshape(1, -1), gt_ps.reshape(1, -1), filter_std=3
        )
        errors.append(float(err))
    return float(np.mean(errors))


def score_spectrum_windows(
    model,
    windows: list[np.ndarray],
    tf_steps: int,
    observation_process: str | None,
    device: str,
) -> float:
    """Teacher-force ``tf_steps``, free-run the rest, score that tail only."""
    predicted = []
    ground_truth = []
    for window in windows:
        arr = np.asarray(window, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        batch = torch.as_tensor(arr, dtype=torch.float32, device=device).unsqueeze(1)
        recon, _selector = run_forward_with_mode(
            model,
            batch,
            mode="flip_at_index",
            observation_process=observation_process,
            flip_point=int(tf_steps),
        )
        predicted.append(recon[:, 0, :].detach().cpu().numpy())
        ground_truth.append(arr)
    return free_run_spectrum_error(predicted, ground_truth, tf_steps)
