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
    dataset_row_group,
    intra_subject_window_starts,
)
from dvae.eval.utils.durstewitz_eval_metrics import power_spectrum_error
from dvae.eval.utils.forward_modes import run_forward_with_mode

_SPLIT_SEED = 1234567890
# Window choice among TF-clean candidates. Fixed so a resume redraws the same set.
VAL_WINDOW_SEED = _SPLIT_SEED
DEFAULT_TF_MIN_OBSERVED = 0.95
DEFAULT_TF_MAX_GAP_FRAC = 0.01


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
    return dataset_row_group(dataset)


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


def longest_missing_run(observed: np.ndarray) -> int:
    """Longest contiguous run of missing (False) steps. Zero when none are missing."""
    missing = ~np.asarray(observed, dtype=bool).reshape(-1)
    if missing.size == 0 or not missing.any():
        return 0
    padded = np.concatenate([[False], missing, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int((ends - starts).max())


def window_tf_stats(window, tf_steps: int) -> tuple[float, int, bool]:
    """Observed fraction and longest missing run on the teacher-forced prefix.

    A step is observed when every channel is finite. The window starts on
    an observed sample when that is true of step 0. An empty prefix (no
    teacher-forced steps) is fully observed with no gap.
    """
    arr = np.asarray(window, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == 0:
        return 0.0, 0, False
    start_observed = bool(np.isfinite(arr[0]).all())
    tf_steps = int(tf_steps)
    if tf_steps <= 0:
        return 1.0, 0, start_observed
    prefix = arr[:tf_steps]
    observed = np.isfinite(prefix).all(axis=1)
    frac = float(observed.mean()) if observed.size else 1.0
    return frac, longest_missing_run(observed), start_observed


def select_tf_clean_windows(
    series,
    starts,
    n_windows: int,
    window_len: int,
    tf_steps: int,
    min_observed: float = DEFAULT_TF_MIN_OBSERVED,
    max_gap_frac: float = DEFAULT_TF_MAX_GAP_FRAC,
    seed: int = VAL_WINDOW_SEED,
) -> tuple[list[dict], bool]:
    """Pick windows whose teacher-forced prefix has no long gap.

    A candidate must already lie inside one segment (``starts`` is that
    list) and must start on an observed sample. It qualifies when the
    prefix observed fraction is at least ``min_observed`` and no missing
    run is longer than ``max_gap_frac * tf_steps``.

    When at least ``n_windows`` qualify, ``n_windows`` of them are drawn
    with ``RandomState(seed)`` and then sorted by start index. When fewer
    qualify, the best-ranked candidates are kept (qualified first, then
    higher observed fraction, then a shorter gap, then an earlier start)
    and ``relaxed`` is true.
    """
    n_windows = int(n_windows)
    window_len = int(window_len)
    tf_steps = int(tf_steps)
    gap_limit = float(max_gap_frac) * float(tf_steps)
    records = []
    for start in np.asarray(starts, dtype=np.int64):
        window = series[int(start) : int(start) + window_len]
        frac, gap, start_observed = window_tf_stats(window, tf_steps)
        if not start_observed:
            continue
        qualified = bool(frac + 1e-12 >= float(min_observed) and gap <= gap_limit + 1e-9)
        records.append(
            {
                "start": int(start),
                "observed_frac": float(frac),
                "max_gap": int(gap),
                "qualified": qualified,
            }
        )
    if not records:
        raise ValueError(
            "validation windows need at least one window of length "
            f"{window_len} that stays inside one segment and starts on "
            "an observed sample."
        )
    qualified = [record for record in records if record["qualified"]]
    relaxed = len(qualified) < n_windows
    if not relaxed:
        rng = np.random.RandomState(int(seed))
        pick = rng.choice(len(qualified), size=n_windows, replace=False)
        chosen = [qualified[int(i)] for i in pick]
    else:
        ranked = sorted(
            records,
            key=lambda record: (
                0 if record["qualified"] else 1,
                -record["observed_frac"],
                record["max_gap"],
                record["start"],
            ),
        )
        chosen = ranked[:n_windows]
        details = ", ".join(
            "start={start} observed={observed_frac:.3f} max_gap={max_gap} "
            "qualified={qualified}".format(**record)
            for record in chosen
        )
        warnings.warn(
            f"Validation windows asked for {n_windows} with TF observed "
            f"fraction >= {float(min_observed)} and no missing run longer "
            f"than {float(max_gap_frac)} of tf_steps ({tf_steps}); only "
            f"{len(qualified)} qualified. Using the best-ranked "
            f"{len(chosen)}: {details}",
            UserWarning,
            stacklevel=2,
        )
    chosen.sort(key=lambda record: record["start"])
    return chosen, relaxed


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


def _window_info(records, relaxed, min_observed, max_gap_frac, seed, tf_steps) -> dict:
    return {
        "starts": np.asarray([record["start"] for record in records], dtype=np.int64),
        "tf_observed": np.asarray(
            [record["observed_frac"] for record in records], dtype=np.float64
        ),
        "tf_max_gap": np.asarray([record["max_gap"] for record in records], dtype=np.int64),
        "qualified": np.asarray([record["qualified"] for record in records], dtype=bool),
        "relaxed": bool(relaxed),
        "min_observed": float(min_observed),
        "max_gap_frac": float(max_gap_frac),
        "seed": int(seed),
        "tf_steps": int(tf_steps),
    }


def capture_fixed_spectrum_windows(
    dataset,
    n_windows: int = 10,
    window_len: int = 2000,
    tf_steps: int | None = None,
    min_observed: float = DEFAULT_TF_MIN_OBSERVED,
    max_gap_frac: float = DEFAULT_TF_MAX_GAP_FRAC,
    seed: int = VAL_WINDOW_SEED,
    return_info: bool = False,
):
    """Copy fixed windows from the validation series.

    Without ``tf_steps``, starts are evenly spaced among windows that fit
    inside one segment. With ``tf_steps``, the teacher-forced prefix must
    be dense: observed fraction >= ``min_observed``, no missing run longer
    than ``max_gap_frac`` of ``tf_steps``, the window inside one segment,
    and the first sample observed. The draw uses ``seed`` and is done once
    by the caller. Fewer than ``n_windows`` qualifiers relaxes to the
    best-ranked windows and warns, with each window's stats in the message.

    Raises when none fit. Does not call ``update_sequence_length``.
    ``return_info`` true also returns the chosen starts and TF stats.
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
            "spectrum validation needs at least one validation window of "
            f"length {window_len}; none fit inside a subject or stay."
        )
    series = _seq_array(dataset)
    if tf_steps is None:
        chosen = select_fixed_starts(starts, n_windows)
        if chosen.size < n_windows:
            warnings.warn(
                f"Spectrum validation asked for {n_windows} windows of length "
                f"{window_len}; using all {chosen.size} that fit.",
                UserWarning,
                stacklevel=2,
            )
        windows = [
            np.array(series[int(s) : int(s) + window_len], copy=True) for s in chosen
        ]
        if return_info:
            return windows, None
        return windows
    records, relaxed = select_tf_clean_windows(
        series,
        starts,
        n_windows,
        window_len,
        int(tf_steps),
        min_observed=min_observed,
        max_gap_frac=max_gap_frac,
        seed=seed,
    )
    windows = [
        np.array(series[int(record["start"]) : int(record["start"]) + window_len], copy=True)
        for record in records
    ]
    info = _window_info(
        records, relaxed, min_observed, max_gap_frac, seed, int(tf_steps)
    )
    if return_info:
        return windows, info
    return windows


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
