"""Original vs model-output spectra during training.

Teacher-forced steps and autonomous steps are drawn separately. Each is
compared with the original on those same steps. ``half_half`` (and a
missing selector) uses the first half against the second half. Other
modes follow the autonomous mode selector, including non-contiguous
bursts: the selected steps are concatenated in time order and are not
interpolated.

The sampling rate is the dataset rate in Hz. ``[DataFrame] sample_rate``
is a downsampling factor and is not used here. An unknown rate labels
the axis in cycles/step.

Missing samples stay NaN in the inputs. For the periodogram the ground
truth's gaps are applied to the model output as well, then each series
is centered on its own finite samples and zero-filled at those gaps.
Channels that fail the shared finiteness rule are left out. Nothing is
resampled.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from dvae.eval.utils.spectrum_validation import (
    MIN_FINITE_FRAC,
    MIN_FINITE_SAMPLES,
    apply_gt_missing_mask,
    segment_has_enough_finite,
    shared_mask_hann_pair,
)
from dvae.timescale_history import cell_A_sigmas, diagonal_A_from_sigmas


def autonomous_time_slice(
    length: int,
    mode_selector=None,
) -> slice:
    """Contiguous autonomous span, or the second half when none is marked.

    ``mode_selector`` is (T,) or (T, C). A step is autonomous when every
    channel at that step is > 0.5. The span runs from the first such step
    through the last. An empty mark falls back to ``[length // 2 :)``.

    Burst modes should not use this span: the spectrum plot concatenates
    the autonomous steps instead of filling the gaps between bursts.
    """
    length = int(length)
    if length < 2:
        return slice(0, length)
    if mode_selector is None:
        return slice(length // 2, length)
    sel = np.asarray(mode_selector)
    if sel.ndim == 1:
        auto = sel.reshape(-1) > 0.5
    else:
        flat = sel.reshape(sel.shape[0], -1)
        auto = (flat > 0.5).all(axis=1)
    idx = np.flatnonzero(auto)
    if idx.size == 0:
        return slice(length // 2, length)
    return slice(int(idx[0]), int(idx[-1]) + 1)


def dataset_sampling_freq_hz(dataset):
    """Dataset sampling frequency in Hz, or None when it is unknown.

    ``None``, non-finite, and non-positive values mean the spectrum axis
    stays in cycles/step. This is not the DataFrame downsampling factor.
    """
    freq = getattr(dataset, "sampling_freq", None)
    if freq is None:
        return None
    try:
        freq = float(freq)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(freq) or freq <= 0:
        return None
    return freq


def timescale_alphas_from_model(model) -> np.ndarray:
    """Current learned alphas, or an empty array when the model has none.

    MT_RNN / MT_VRNN use ``model.sigmas``. PLRNN / shPLRNN use the
    recurrent cell's ``A_sigmas``. Both go through the base-10 sigmoid.
    Non-finite and non-positive alphas are dropped. Vanilla RNN and
    shPLRNN without ``A`` return an empty array so the plot draws no lines.
    """
    if model is None:
        return np.zeros(0, dtype=np.float64)
    values = None
    sigmas = getattr(model, "sigmas", None)
    if torch.is_tensor(sigmas) and sigmas.numel() > 0:
        values = sigmas.detach().cpu().numpy().reshape(-1)
    else:
        cell = cell_A_sigmas(model)
        if torch.is_tensor(cell) and cell.numel() > 0:
            values = cell.detach().cpu().numpy().reshape(-1)
    if values is None:
        return np.zeros(0, dtype=np.float64)
    alphas = diagonal_A_from_sigmas(values)
    keep = np.isfinite(alphas) & (alphas > 0)
    return np.asarray(alphas[keep], dtype=np.float64)


def alpha_line_frequencies(alphas, sample_rate) -> np.ndarray:
    """Timescale lines at ``f = alpha / dt``.

    ``dt`` is ``1 / sampling_freq`` when the rate is known, so
    ``f = alpha * sampling_freq`` in Hz. An unknown rate leaves ``f``
    in cycles/step, equal to alpha.
    """
    alphas = np.asarray(alphas, dtype=np.float64).reshape(-1)
    alphas = alphas[np.isfinite(alphas) & (alphas > 0)]
    rate, _unit = _rate_and_unit(sample_rate)
    if rate is None:
        return alphas
    return alphas * rate


def _rate_and_unit(sample_rate):
    if sample_rate is None:
        return None, "cycles/step"
    try:
        rate = float(sample_rate)
    except (TypeError, ValueError):
        return None, "cycles/step"
    if not np.isfinite(rate) or rate <= 0:
        return None, "cycles/step"
    return rate, "Hz"


def _as_batch(original, output):
    """``(batch, time, channel)``. 2D is one sequence; 1D is one channel."""
    orig = np.asarray(original, dtype=np.float64)
    out = np.asarray(output, dtype=np.float64)
    if orig.shape != out.shape:
        raise ValueError(
            "Original and output must share shape; "
            f"got {orig.shape} vs {out.shape}."
        )
    if orig.ndim == 1:
        orig = orig.reshape(1, -1, 1)
        out = out.reshape(1, -1, 1)
    elif orig.ndim == 2:
        orig = orig.reshape(1, orig.shape[0], orig.shape[1])
        out = out.reshape(1, out.shape[0], out.shape[1])
    elif orig.ndim != 3:
        raise ValueError(
            "Spectrum comparison expects (time,), (time, channel), or "
            f"(batch, time, channel); got {orig.shape}."
        )
    return orig, out


def _broadcast_selector(selector, n_seq, n_time, n_ch):
    """Selector as (batch, time, channel). None is half teacher-forced."""
    if selector is None:
        sel = np.zeros((n_seq, n_time, n_ch), dtype=np.float64)
        sel[:, n_time // 2 :, :] = 1.0
        return sel
    sel = np.asarray(selector, dtype=np.float64)
    if sel.ndim == 1:
        if sel.shape != (n_time,):
            raise ValueError(
                f"mode_selector length {sel.shape[0]} != time {n_time}."
            )
        return np.broadcast_to(sel.reshape(1, n_time, 1), (n_seq, n_time, n_ch)).copy()
    if sel.ndim == 2:
        if sel.shape == (n_seq, n_time):
            return np.broadcast_to(sel[:, :, None], (n_seq, n_time, n_ch)).copy()
        if sel.shape == (n_time, n_ch):
            return np.broadcast_to(sel[None, :, :], (n_seq, n_time, n_ch)).copy()
        raise ValueError(
            f"mode_selector shape {sel.shape} is not (batch, time) or "
            f"(time, channel) for batch {n_seq}, time {n_time}, "
            f"channels {n_ch}."
        )
    if sel.ndim == 3:
        if sel.shape != (n_seq, n_time, n_ch):
            raise ValueError(
                f"mode_selector shape {sel.shape} != "
                f"({n_seq}, {n_time}, {n_ch})."
            )
        return sel
    raise ValueError(f"mode_selector ndim {sel.ndim} is not supported.")


def _segment_indices(selector_1d, kind: str) -> np.ndarray:
    """Time indices kept for one segment, in order. Gaps are not filled."""
    sel = np.asarray(selector_1d, dtype=np.float64).reshape(-1)
    if kind == "autonomous":
        keep = sel > 0.5
    elif kind == "tf":
        keep = ~(sel > 0.5)
    else:
        raise ValueError(f"segment kind must be 'tf' or 'autonomous', got {kind!r}.")
    return np.flatnonzero(keep)


def _pick_length(lengths) -> Optional[int]:
    """Most common length. A tie keeps the longer one."""
    if not lengths:
        return None
    best_len = None
    best_count = -1
    for length in sorted(set(lengths), reverse=True):
        count = sum(1 for item in lengths if item == length)
        if count > best_count:
            best_count = count
            best_len = int(length)
    return best_len


def _freqs_for(length: int, rate) -> np.ndarray:
    if length < 2:
        return np.zeros(0, dtype=np.float64)
    spacing = 1.0 if rate is None else 1.0 / float(rate)
    return np.fft.rfftfreq(int(length), d=spacing)[1:]


def _pad_rows(rows, width: int) -> np.ndarray:
    out = np.full((len(rows), width), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        if row is None or len(row) == 0:
            continue
        out[index, : len(row)] = row
    return out


def _mean_finite(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def compute_original_vs_output_spectra(
    original,
    output,
    *,
    mode_selector=None,
    sample_rate=None,
    min_finite: int = MIN_FINITE_SAMPLES,
    min_frac: float = MIN_FINITE_FRAC,
) -> dict:
    """Teacher-forced and autonomous Hann PSDs, averaged over the batch.

    Each sequence is split by the mode selector. A channel is included
    when at least one sequence passes the finiteness rule on that
    segment. PSDs are averaged only among sequences whose segment length
    is the most common one for that channel (ties keep the longer
    length). Different lengths are not resampled onto one grid.
    """
    orig, out = _as_batch(original, output)
    n_seq, n_time, n_channels = orig.shape
    selector = _broadcast_selector(mode_selector, n_seq, n_time, n_channels)
    rate, unit = _rate_and_unit(sample_rate)

    result = {
        "n_sequences": int(n_seq),
        "n_channels": int(n_channels),
        "sample_rate": float("nan") if rate is None else float(rate),
        "frequency_unit": unit,
    }
    for kind in ("tf", "autonomous"):
        included = np.zeros(n_channels, dtype=bool)
        averaged = np.zeros(n_channels, dtype=np.int64)
        seg_lengths = np.zeros(n_channels, dtype=np.int64)
        miss_per_channel = np.full(n_channels, np.nan, dtype=np.float64)
        orig_rows = [None] * n_channels
        out_rows = [None] * n_channels
        freq_rows = [None] * n_channels
        max_bins = 0
        for channel in range(n_channels):
            spectra = []
            missing = []
            for seq in range(n_seq):
                index = _segment_indices(selector[seq, :, channel], kind)
                if index.size == 0:
                    continue
                gt_seg = orig[seq, index, channel]
                pred_seg = out[seq, index, channel]
                missing.append(float(1.0 - np.isfinite(gt_seg).mean()))
                # The ground-truth gaps are applied to the prediction
                # before the skip rule, so a value that exists only inside
                # a ground-truth gap does not keep the channel.
                gt_masked, pred_masked = apply_gt_missing_mask(gt_seg, pred_seg)
                if not (
                    segment_has_enough_finite(gt_masked, min_finite, min_frac)
                    and segment_has_enough_finite(pred_masked, min_finite, min_frac)
                ):
                    continue
                gt_psd, pred_psd = shared_mask_hann_pair(gt_seg, pred_seg)
                spectra.append((seq, gt_psd, pred_psd, int(gt_seg.shape[0])))
            miss_per_channel[channel] = _mean_finite(missing)
            if not spectra:
                continue
            chosen_len = _pick_length([item[3] for item in spectra])
            chosen = [item for item in spectra if item[3] == chosen_len]
            if not chosen:
                continue
            orig_mean = np.mean([item[1] for item in chosen], axis=0)
            out_mean = np.mean([item[2] for item in chosen], axis=0)
            freqs = _freqs_for(chosen_len, rate)
            if freqs.shape[0] != orig_mean.shape[0]:
                raise RuntimeError(
                    "Shared-mask periodogram and frequency grid disagree: "
                    f"{orig_mean.shape[0]} bins vs {freqs.shape[0]}."
                )
            included[channel] = True
            averaged[channel] = int(len({item[0] for item in chosen}))
            seg_lengths[channel] = int(chosen_len)
            orig_rows[channel] = orig_mean
            out_rows[channel] = out_mean
            freq_rows[channel] = freqs
            max_bins = max(max_bins, int(freqs.shape[0]))
        n_excluded = int((~included).sum()) if n_channels else 0
        result[f"channel_included_{kind}"] = included
        result[f"n_sequences_averaged_{kind}"] = averaged
        result[f"segment_length_{kind}"] = seg_lengths
        result[f"missing_sample_fraction_{kind}_per_channel"] = miss_per_channel
        result[f"missing_sample_fraction_{kind}"] = _mean_finite(
            miss_per_channel[included] if included.any() else miss_per_channel
        )
        result[f"excluded_channel_fraction_{kind}"] = (
            float(n_excluded) / float(n_channels) if n_channels else 0.0
        )
        result[f"n_excluded_channels_{kind}"] = n_excluded
        result[f"frequencies_{kind}"] = _pad_rows(freq_rows, max_bins)
        result[f"original_psd_{kind}"] = _pad_rows(orig_rows, max_bins)
        result[f"output_psd_{kind}"] = _pad_rows(out_rows, max_bins)
    return result


def spectrum_comparison_paths(save_dir: str, explain: str) -> tuple[str, str]:
    """PNG and NPZ paths under ``save_dir`` for one spectrum comparison."""
    os.makedirs(save_dir, exist_ok=True)
    base = f"spectrum_compare_{explain}" if explain else "spectrum_compare"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in base)
    return (
        os.path.join(save_dir, f"{safe}.png"),
        os.path.join(save_dir, f"{safe}.npz"),
    )


def _display_psd(psd) -> np.ndarray:
    """Drop non-positive bins so a log axis does not warn. NPZ keeps raw power."""
    shown = np.asarray(psd, dtype=np.float64).copy()
    shown[~(np.isfinite(shown) & (shown > 0))] = np.nan
    return shown


def _finite_prefix(freqs, psd) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    psd = np.asarray(psd, dtype=np.float64).reshape(-1)
    n = min(freqs.size, psd.size)
    if n == 0:
        return freqs[:0], psd[:0]
    finite_freq = np.isfinite(freqs[:n])
    if not finite_freq.any():
        return freqs[:0], psd[:0]
    last = int(np.flatnonzero(finite_freq)[-1]) + 1
    return freqs[:last], psd[:last]


def _column_count(averaged, included) -> tuple[int, bool]:
    """How many sequences entered the average, and whether that count is shared."""
    vals = [
        int(averaged[index])
        for index in range(len(included))
        if included[index]
    ]
    if not vals:
        return 0, True
    return int(max(vals)), len(set(vals)) == 1


def _pct(frac) -> str:
    if frac is None or not np.isfinite(frac):
        return "n/a"
    return f"{100.0 * float(frac):.1f}%"


def _draw_segment(ax, freqs, original_psd, output_psd, included, alpha_freqs):
    """One panel. Returns whether a timescale line was drawn inside the axes."""
    drew_alpha = False
    if not included:
        ax.text(
            0.5,
            0.5,
            "Excluded\n(too few finite samples)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="0.35",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return drew_alpha
    freq, orig = _finite_prefix(freqs, original_psd)
    _, out = _finite_prefix(freqs, output_psd)
    orig = _display_psd(orig)
    out = _display_psd(out)
    if freq.size and (np.isfinite(orig).any() or np.isfinite(out).any()):
        ax.semilogy(freq, orig, color="#1f77b4", lw=1.5, label="Original")
        ax.semilogy(freq, out, color="#ff7f0e", lw=1.5, label="Model output")
        ax.grid(True, which="both", axis="y", alpha=0.25, lw=0.4)
        xmin, xmax = ax.get_xlim()
        for freq_line in np.asarray(alpha_freqs, dtype=np.float64).reshape(-1):
            if not np.isfinite(freq_line) or freq_line <= 0:
                continue
            if xmin < float(freq_line) < xmax:
                ax.axvline(
                    float(freq_line),
                    color="0.45",
                    lw=0.7,
                    alpha=0.45,
                    zorder=0,
                )
                drew_alpha = True
        ax.set_xlim(xmin, xmax)
    else:
        ax.text(
            0.5,
            0.5,
            "No positive power",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="0.35",
        )
    return drew_alpha


def save_original_vs_output_spectrum(
    original,
    output,
    save_dir: str,
    *,
    explain: str = "",
    mode_selector=None,
    sample_rate=None,
    epoch: Optional[int] = None,
    mode_name: str = "",
    alpha_frequencies=None,
) -> dict:
    """Plot teacher-forced and autonomous spectra and save the numeric values.

    Two columns: teacher-forced and autonomous. Each column is the
    original and the model output on the same steps. Faint vertical
    lines mark model timescale frequencies when ``alpha_frequencies``
    is provided. Lines outside the data range are not drawn and do not
    widen the axes.
    """
    result = compute_original_vs_output_spectra(
        original,
        output,
        mode_selector=mode_selector,
        sample_rate=sample_rate,
    )
    if alpha_frequencies is None:
        alpha_frequencies = np.zeros(0, dtype=np.float64)
    alpha_frequencies = np.asarray(alpha_frequencies, dtype=np.float64).reshape(-1)
    alpha_frequencies = alpha_frequencies[
        np.isfinite(alpha_frequencies) & (alpha_frequencies > 0)
    ]
    png_path, npz_path = spectrum_comparison_paths(save_dir, explain)
    n_channels = int(result["n_channels"])
    n_plot = max(n_channels, 1)
    unit = result["frequency_unit"]
    fig, axes = plt.subplots(
        n_plot,
        2,
        figsize=(10.0, max(3.2, 2.4 * n_plot)),
        squeeze=False,
    )
    drew_alpha = False
    for channel in range(n_plot):
        for column, kind in enumerate(("tf", "autonomous")):
            ax = axes[channel, column]
            if channel >= n_channels:
                ax.set_visible(False)
                continue
            drew_alpha = _draw_segment(
                ax,
                result[f"frequencies_{kind}"][channel],
                result[f"original_psd_{kind}"][channel],
                result[f"output_psd_{kind}"][channel],
                bool(result[f"channel_included_{kind}"][channel]),
                alpha_frequencies,
            ) or drew_alpha
            if column == 0:
                ax.set_ylabel(f"Channel {channel}\nPSD")
            if channel == n_channels - 1:
                ax.set_xlabel(f"Frequency ({unit})")
    tf_n, tf_same = _column_count(
        result["n_sequences_averaged_tf"], result["channel_included_tf"]
    )
    auto_n, auto_same = _column_count(
        result["n_sequences_averaged_autonomous"],
        result["channel_included_autonomous"],
    )
    tf_title = (
        f"Teacher-forced, {tf_n} sequence{'s' if tf_n != 1 else ''} averaged"
        if tf_same
        else f"Teacher-forced, up to {tf_n} sequences averaged"
    )
    auto_title = (
        f"Autonomous, {auto_n} sequence{'s' if auto_n != 1 else ''} averaged"
        if auto_same
        else f"Autonomous, up to {auto_n} sequences averaged"
    )
    axes[0, 0].set_title(tf_title)
    axes[0, 1].set_title(auto_title)
    title_bits = ["Teacher-forced and autonomous spectra vs original"]
    if mode_name:
        title_bits.append(f"({mode_name})")
    if epoch is not None:
        title_bits.append(f"epoch {int(epoch)}")
    fig.suptitle(" ".join(title_bits), fontsize=12)
    handles = [
        Line2D([0], [0], color="#1f77b4", lw=1.6, label="Original"),
        Line2D([0], [0], color="#ff7f0e", lw=1.6, label="Model output"),
    ]
    if drew_alpha:
        handles.append(
            Line2D(
                [0],
                [0],
                color="0.45",
                lw=0.9,
                alpha=0.8,
                label="Model timescale (α/dt)",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
    )
    annotation = (
        f"Batch sequences: {result['n_sequences']}. "
        f"Excluded channels: teacher-forced "
        f"{result['n_excluded_channels_tf']}/{n_channels} "
        f"({_pct(result['excluded_channel_fraction_tf'])}), "
        f"autonomous {result['n_excluded_channels_autonomous']}/{n_channels} "
        f"({_pct(result['excluded_channel_fraction_autonomous'])}). "
        f"Missing samples: teacher-forced "
        f"{_pct(result['missing_sample_fraction_tf'])}, "
        f"autonomous {_pct(result['missing_sample_fraction_autonomous'])}."
    )
    fig.text(0.5, 0.005, annotation, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.88))
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        npz_path,
        frequencies_tf=result["frequencies_tf"],
        frequencies_autonomous=result["frequencies_autonomous"],
        original_psd_tf=result["original_psd_tf"],
        output_psd_tf=result["output_psd_tf"],
        original_psd_autonomous=result["original_psd_autonomous"],
        output_psd_autonomous=result["output_psd_autonomous"],
        alpha_frequencies=np.asarray(alpha_frequencies, dtype=np.float64),
        channel_included_tf=result["channel_included_tf"],
        channel_included_autonomous=result["channel_included_autonomous"],
        excluded_channel_fraction_tf=np.asarray(
            result["excluded_channel_fraction_tf"], dtype=np.float64
        ),
        excluded_channel_fraction_autonomous=np.asarray(
            result["excluded_channel_fraction_autonomous"], dtype=np.float64
        ),
        missing_sample_fraction_tf=np.asarray(
            result["missing_sample_fraction_tf"], dtype=np.float64
        ),
        missing_sample_fraction_autonomous=np.asarray(
            result["missing_sample_fraction_autonomous"], dtype=np.float64
        ),
        missing_sample_fraction_tf_per_channel=result[
            "missing_sample_fraction_tf_per_channel"
        ],
        missing_sample_fraction_autonomous_per_channel=result[
            "missing_sample_fraction_autonomous_per_channel"
        ],
        n_sequences=np.asarray(result["n_sequences"], dtype=np.int64),
        n_sequences_averaged_tf=result["n_sequences_averaged_tf"],
        n_sequences_averaged_autonomous=result["n_sequences_averaged_autonomous"],
        segment_length_tf=result["segment_length_tf"],
        segment_length_autonomous=result["segment_length_autonomous"],
        sample_rate=np.asarray(result["sample_rate"], dtype=np.float64),
        frequency_unit=np.asarray(unit),
        epoch=np.asarray(-1 if epoch is None else int(epoch), dtype=np.int64),
        mode_name=np.asarray(mode_name or ""),
        explain=np.asarray(explain or ""),
    )
    result["png_path"] = png_path
    result["npz_path"] = npz_path
    result["epoch"] = epoch
    result["mode_name"] = mode_name
    result["alpha_frequencies"] = alpha_frequencies
    return result
