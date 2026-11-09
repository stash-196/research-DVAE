"""Original vs model-output spectrum plots during training.

Drawn whenever the sequence-reconstruction waveform figures are saved.
Reuses the Hann periodogram and finiteness rules from
``spectrum_validation``: a channel is excluded when it has fewer than
``MIN_FINITE_SAMPLES`` finite points or under ``MIN_FINITE_FRAC`` finite.
Missing samples stay NaN; nothing is filled, resampled, or interpolated.

For ``half_half`` the spectrum uses the autonomous second half. For the
other modes the full sequence is used (cheap and matches the waveform
arrays). Figures are paper-ready; numeric arrays are saved alongside.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dvae.eval.utils.spectrum_validation import (
    MIN_FINITE_FRAC,
    MIN_FINITE_SAMPLES,
    _hann_periodogram,
    segment_has_enough_finite,
)


def autonomous_time_slice(
    length: int,
    mode_selector=None,
) -> slice:
    """Contiguous autonomous span, or the second half when none is marked.

    ``mode_selector`` is (T,) or (T, C). A step is autonomous when every
    channel at that step is > 0.5. The span runs from the first such step
    through the last. An empty mark falls back to ``[length // 2 :)``.
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


def _as_2d(series) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            "Spectrum comparison needs a (time,) or (time, channel) series, "
            f"got shape {arr.shape}."
        )
    return arr


def compute_original_vs_output_spectra(
    original,
    output,
    *,
    mode_selector=None,
    region: str = "full",
    sample_rate: float = 1.0,
    min_finite: int = MIN_FINITE_SAMPLES,
    min_frac: float = MIN_FINITE_FRAC,
) -> dict:
    """Per-channel Hann PSD of original and model output from the same arrays.

    ``region`` is ``\"autonomous\"`` (prefer the autonomous span) or
    ``\"full\"``. Channels that fail the finiteness check on either the
    original or the output segment are excluded; their PSD rows are NaN.
    Nothing is filled or resampled.

    Returns a dict with frequencies, PSD arrays, exclusion stats, and the
    time slice used. Sample rate is in samples per second; frequency is Hz.
    """
    orig = _as_2d(original)
    out = _as_2d(output)
    if orig.shape != out.shape:
        raise ValueError(
            "Original and output must share shape; "
            f"got {orig.shape} vs {out.shape}."
        )
    n_time, n_channels = orig.shape
    if region == "autonomous":
        time_slice = autonomous_time_slice(n_time, mode_selector)
    elif region == "full":
        time_slice = slice(0, n_time)
    else:
        raise ValueError(f"region must be 'autonomous' or 'full', got {region!r}.")

    orig_seg = orig[time_slice]
    out_seg = out[time_slice]
    seg_len = int(orig_seg.shape[0])
    if seg_len < 2:
        freqs = np.asarray([], dtype=np.float64)
        nan_psd = np.full((n_channels, 0), np.nan, dtype=np.float64)
        return {
            "frequencies_hz": freqs,
            "original_psd": nan_psd,
            "output_psd": nan_psd,
            "channel_included": np.zeros(n_channels, dtype=bool),
            "excluded_channel_fraction": 1.0 if n_channels else 0.0,
            "missing_sample_fraction": 1.0,
            "n_channels": n_channels,
            "n_excluded_channels": n_channels,
            "segment_length": seg_len,
            "time_slice_start": int(time_slice.start or 0),
            "time_slice_stop": int(time_slice.stop or 0),
            "region": region,
            "sample_rate": float(sample_rate),
        }

    freqs = np.fft.rfftfreq(seg_len, d=1.0 / float(sample_rate))[1:]
    original_psd = np.full((n_channels, freqs.size), np.nan, dtype=np.float64)
    output_psd = np.full((n_channels, freqs.size), np.nan, dtype=np.float64)
    included = np.zeros(n_channels, dtype=bool)
    missing_fracs = []

    for channel in range(n_channels):
        o = orig_seg[:, channel]
        p = out_seg[:, channel]
        o_ok = segment_has_enough_finite(o, min_finite, min_frac)
        p_ok = segment_has_enough_finite(p, min_finite, min_frac)
        finite_both = np.isfinite(o) & np.isfinite(p)
        missing_fracs.append(1.0 - float(finite_both.mean()) if o.size else 1.0)
        if not (o_ok and p_ok):
            continue
        original_psd[channel] = _hann_periodogram(o)
        output_psd[channel] = _hann_periodogram(p)
        included[channel] = True

    n_excluded = int((~included).sum())
    return {
        "frequencies_hz": freqs,
        "original_psd": original_psd,
        "output_psd": output_psd,
        "channel_included": included,
        "excluded_channel_fraction": (
            float(n_excluded) / float(n_channels) if n_channels else 0.0
        ),
        "missing_sample_fraction": (
            float(np.mean(missing_fracs)) if missing_fracs else 1.0
        ),
        "n_channels": n_channels,
        "n_excluded_channels": n_excluded,
        "segment_length": seg_len,
        "time_slice_start": int(time_slice.start or 0),
        "time_slice_stop": int(time_slice.stop or seg_len),
        "region": region,
        "sample_rate": float(sample_rate),
    }


def spectrum_comparison_paths(save_dir: str, explain: str) -> tuple[str, str]:
    """PNG and NPZ paths under ``save_dir`` for one spectrum comparison."""
    os.makedirs(save_dir, exist_ok=True)
    base = f"spectrum_compare_{explain}" if explain else "spectrum_compare"
    # Keep filenames filesystem-safe.
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in base)
    return (
        os.path.join(save_dir, f"{safe}.png"),
        os.path.join(save_dir, f"{safe}.npz"),
    )


def save_original_vs_output_spectrum(
    original,
    output,
    save_dir: str,
    *,
    explain: str = "",
    mode_selector=None,
    region: str = "full",
    sample_rate: float = 1.0,
    epoch: Optional[int] = None,
    mode_name: str = "",
) -> dict:
    """Compute, plot, and save original vs output spectra plus numeric values.

    Writes a paper-ready PNG (log-scale PSD) and an NPZ with frequencies,
    both PSDs, exclusion stats, epoch, and region. Returns the paths and
    the compute dict.
    """
    result = compute_original_vs_output_spectra(
        original,
        output,
        mode_selector=mode_selector,
        region=region,
        sample_rate=sample_rate,
    )
    png_path, npz_path = spectrum_comparison_paths(save_dir, explain)

    freqs = result["frequencies_hz"]
    orig_psd = result["original_psd"]
    out_psd = result["output_psd"]
    included = result["channel_included"]
    n_channels = result["n_channels"]
    n_excluded = result["n_excluded_channels"]
    excl_frac = result["excluded_channel_fraction"]
    miss_frac = result["missing_sample_fraction"]

    n_plot = max(int(n_channels), 1)
    fig_h = max(2.8, 2.2 * n_plot)
    fig, axes = plt.subplots(n_plot, 1, figsize=(7.5, fig_h), sharex=True)
    if n_plot == 1:
        axes = [axes]

    region_label = (
        "autonomous segment" if result["region"] == "autonomous" else "full sequence"
    )
    for channel, ax in enumerate(axes):
        if channel >= n_channels:
            ax.set_visible(False)
            continue
        if included[channel] and freqs.size:
            ax.semilogy(
                freqs,
                orig_psd[channel],
                color="C0",
                lw=1.4,
                label="Original",
            )
            ax.semilogy(
                freqs,
                out_psd[channel],
                color="C1",
                lw=1.4,
                label="Model output",
            )
            ax.set_ylabel(f"PSD ch {channel}")
        else:
            ax.text(
                0.5,
                0.5,
                "Excluded (too few finite samples)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.set_ylabel(f"PSD ch {channel}")
        if channel == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=9)

    axes[-1].set_xlabel("Frequency (Hz)")
    title_bits = ["Original vs output spectrum"]
    if mode_name:
        title_bits.append(f"({mode_name}, {region_label})")
    else:
        title_bits.append(f"({region_label})")
    if epoch is not None:
        title_bits.append(f"epoch {int(epoch)}")
    fig.suptitle(" ".join(title_bits), fontsize=12)
    annotation = (
        f"Excluded channels: {n_excluded}/{n_channels} "
        f"({100.0 * excl_frac:.1f}%); "
        f"missing samples in segment: {100.0 * miss_frac:.1f}%"
    )
    fig.text(0.5, 0.01, annotation, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        npz_path,
        frequencies_hz=result["frequencies_hz"],
        original_psd=result["original_psd"],
        output_psd=result["output_psd"],
        channel_included=result["channel_included"],
        excluded_channel_fraction=np.asarray(
            result["excluded_channel_fraction"], dtype=np.float64
        ),
        missing_sample_fraction=np.asarray(
            result["missing_sample_fraction"], dtype=np.float64
        ),
        n_channels=np.asarray(result["n_channels"], dtype=np.int64),
        n_excluded_channels=np.asarray(result["n_excluded_channels"], dtype=np.int64),
        segment_length=np.asarray(result["segment_length"], dtype=np.int64),
        time_slice_start=np.asarray(result["time_slice_start"], dtype=np.int64),
        time_slice_stop=np.asarray(result["time_slice_stop"], dtype=np.int64),
        sample_rate=np.asarray(result["sample_rate"], dtype=np.float64),
        epoch=np.asarray(-1 if epoch is None else int(epoch), dtype=np.int64),
        region=np.asarray(result["region"]),
        mode_name=np.asarray(mode_name or ""),
        explain=np.asarray(explain or ""),
    )

    result["png_path"] = png_path
    result["npz_path"] = npz_path
    result["epoch"] = epoch
    result["mode_name"] = mode_name
    return result
