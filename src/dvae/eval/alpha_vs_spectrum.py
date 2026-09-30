"""Eval figure: timescale trajectories beside the signal power spectrum.

MT_RNN / MT_VRNN historically drew this from ``loss_model.pckl``
``sigmas_history`` inside spectrum analysis, converting logits with a
natural sigmoid. The call was gated on the MT model name, the spectrum
panel title was hardcoded to Lorenz63, and the spectra themselves came
from a dataset if/elif (Lorenz63, SHO, DampedSHO, Xhro) that did not
include XhroProper or XhroPacketLoss. After the benchmark rewrite the
call sat below a ``return`` and was then deleted, so no dataset wrote
the figure. This module draws it for whatever dataset produced the
channel benchmarks, including XHRO, whenever a timescale history exists.
PLRNN / shPLRNN use ``A_sigmas`` (diagonal A via the cell's base-10
sigmoid).
"""

import os

import numpy as np

from dvae.timescale_history import MT_MODEL_NAMES, diagonal_A_from_sigmas


def eval_timescale_tracks(loaded_data, model_name):
    """Return plot tracks present in a loss pickle.

    Each track is ``{"history", "kind", "explain"}``.
    ``kind="mt"`` keeps the historical natural-sigmoid figure.
    ``kind="plrnn"`` is diagonal A. When both exist, the PLRNN file uses
    explain ``diagA`` so it does not replace the MT figure.
    """
    if not loaded_data:
        return []
    is_mt = model_name in MT_MODEL_NAMES
    tracks = []
    if is_mt and "sigmas_history" in loaded_data:
        tracks.append(
            {
                "history": np.asarray(loaded_data["sigmas_history"], dtype=np.float64),
                "kind": "mt",
                "explain": "",
            }
        )
    a_hist = loaded_data.get("A_sigmas_history")
    if a_hist is None and not is_mt and "sigmas_history" in loaded_data:
        a_hist = loaded_data["sigmas_history"]
    if a_hist is not None:
        tracks.append(
            {
                "history": np.asarray(a_hist, dtype=np.float64),
                "kind": "plrnn",
                "explain": "diagA" if is_mt else "",
            }
        )
    usable = []
    for track in tracks:
        history = np.asarray(track["history"], dtype=np.float64)
        if history.ndim == 1:
            history = history.reshape(1, -1)
        if history.ndim != 2 or history.size == 0 or history.shape[1] == 0:
            continue
        track = dict(track)
        track["history"] = history
        usable.append(track)
    return usable


def _true_alphas(dataset):
    raw = getattr(dataset, "true_alphas", None)
    if raw is None:
        return []
    values = []
    for item in list(raw):
        value = float(item)
        if value > 0:
            values.append(value)
    return values


def _kl_warm_epochs(loaded_data):
    if not loaded_data:
        return None
    kl = loaded_data.get("kl_warm_epochs")
    if kl is None:
        return None
    kl = np.asarray(kl).reshape(-1)
    if kl.size == 0:
        return None
    return kl


def gt_power_spectra(channel_benchmarks, max_series=8):
    """Positive-frequency spectra of the ground-truth channels.

    Every benchmark channel is included, up to ``max_series`` (XHRO's four
    wearable channels fit; Lorenz's three do too). Series are zero-padded
    to a shared length so they share one frequency grid, matching
    ``visualize_spectral_analysis``. There is no Lorenz-only branch.
    """
    channels = (channel_benchmarks or {}).get("channels") or []
    dt = float((channel_benchmarks or {}).get("dt", 0.0) or 0.0)
    if dt <= 0:
        return None
    series = []
    colors = []
    names = []
    for channel in channels:
        if len(series) >= max_series:
            break
        raw = channel.get("gt_full", channel.get("gt_auto"))
        if raw is None:
            continue
        signal = np.nan_to_num(np.asarray(raw, dtype=np.float64).reshape(-1), nan=0.0)
        if signal.size < 4:
            continue
        series.append(signal)
        colors.append(channel.get("color", "blue"))
        names.append(str(channel.get("name", channel.get("key", "gt"))))
    if not series:
        return None
    max_len = max(len(signal) for signal in series)
    power = []
    frequencies = None
    for signal in series:
        padded = np.pad(signal, (0, max_len - len(signal)))
        spectrum = np.fft.fft(padded)
        freqs = np.fft.fftfreq(max_len, d=dt)
        positive = freqs > 0
        power.append(np.abs(spectrum[positive]) ** 2)
        frequencies = freqs[positive]
    if frequencies is None or frequencies.size == 0:
        return None
    return power, frequencies, colors, names


def save_alpha_vs_spectrum_figures(
    loaded_data,
    model_name,
    channel_benchmarks,
    dataset,
    save_dir,
    visualize_alpha_history_and_spectrums,
    dataset_name=None,
):
    """Write alpha-vs-spectrum figures. Returns the paths that were requested.

    Any dataset is eligible once a timescale history and channel benchmarks
    exist. The spectrum panel is titled with that dataset name. Lorenz MT
    runs still read ``Lorenz63\\nPower Spectrum`` and still use the natural
    sigmoid. PLRNN tracks pass the base-10 diagonal-A transform.
    """
    tracks = eval_timescale_tracks(loaded_data, model_name)
    spectra = gt_power_spectra(channel_benchmarks)
    if not tracks or spectra is None:
        return []
    power, frequencies, colors, names = spectra
    os.makedirs(save_dir, exist_ok=True)
    true_alphas = _true_alphas(dataset)
    kl_warm_epochs = _kl_warm_epochs(loaded_data)
    dt = float(channel_benchmarks["dt"])
    title_name = dataset_name or getattr(dataset, "dataset_name", None) or model_name
    saved = []
    for track in tracks:
        explain = track["explain"]
        kwargs = {
            "sigmas_history": track["history"],
            "power_spectrum_lst": power,
            "spectrum_color_lst": colors,
            "spectrum_name_lst": names,
            "frequencies": frequencies,
            "dt": dt,
            "save_dir": save_dir,
            "kl_warm_epochs": kl_warm_epochs,
            "true_alphas": true_alphas,
            "explain": explain,
            "spectrum_title": f"{title_name}\nPower Spectrum",
        }
        if track["kind"] == "plrnn":
            kwargs["alpha_from_sigma"] = diagonal_A_from_sigmas
            kwargs["max_curves"] = int(track["history"].shape[0])
        visualize_alpha_history_and_spectrums(**kwargs)
        saved.append(
            os.path.join(save_dir, f"vis_alpha_vs_power_spectrum_{explain}.png")
        )
    return saved
