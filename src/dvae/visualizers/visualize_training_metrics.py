import matplotlib.pyplot as plt
import os
import numpy as np

from dvae.eval.aggregate_plot_style import (
    apply_paper_ready_line_style,
    setup_plot_y_axis,
)


def draw_regularizer_warmup_lines(ax, subband_warm_epochs=None, decorr_warm_epochs=None):
    """Vertical markers for subband and decorr scale changes.

    Orange dashed: subband warm-up scale. Purple dotted: decorr warm-up
    scale. ``None`` draws nothing, which is how a disabled term stays
    off the figure. A passed list is the same change-epoch set the
    other warm-up lines use, including the leading 0.
    """
    if subband_warm_epochs is not None:
        for epoch in np.asarray(subband_warm_epochs).reshape(-1):
            ax.axvline(x=float(epoch), color="tab:orange", linestyle=(0, (2, 2)))
    if decorr_warm_epochs is not None:
        for epoch in np.asarray(decorr_warm_epochs).reshape(-1):
            ax.axvline(x=float(epoch), color="tab:purple", linestyle=(0, (1, 1)))


def draw_loss_warmup_lines(
    ax,
    model_name,
    auto_sampling_method,
    noise_sampling_method,
    kl_warm_epochs,
    auto_warm_epochs,
    noise_warm_epochs,
    sequence_len_epochs,
    subband_warm_epochs=None,
    decorr_warm_epochs=None,
):
    """Vertical warmup markers used by the loss-over-epochs figure.

    Cyan dashed: KL warm-up, VRNN / MT_VRNN only. Red dotted: autonomous
    warm-up when the sampler is ss or sm. Green dash-dot: noise warm-up
    on the same samplers. Magenta dash-dot: sequence-length changes.
    Orange and purple mark subband and decorr scale changes when those
    terms are enabled.
    """
    if model_name in ["VRNN", "MT_VRNN"]:
        for kl_warm_epoch in kl_warm_epochs:
            ax.axvline(x=kl_warm_epoch, color="c", linestyle="--")
    if auto_sampling_method in ["ss", "sm"]:
        for auto_warm_epoch in auto_warm_epochs:
            ax.axvline(x=auto_warm_epoch, color="r", linestyle=":")
    if noise_sampling_method in ["ss", "sm"]:
        for noise_warm_epoch in noise_warm_epochs:
            ax.axvline(x=noise_warm_epoch, color="g", linestyle="-.")
    for sequence_len_epoch in sequence_len_epochs:
        ax.axvline(x=sequence_len_epoch, color="m", linestyle="-.")
    draw_regularizer_warmup_lines(ax, subband_warm_epochs, decorr_warm_epochs)


# Four Appendix B terms. Log figures use a symlog axis so a negative
# error_drift_cross stays on the panel. The linear band is the 10th
# percentile of the finite |cross| values on that panel, and at least
# 1e-4. The plotted cross stays signed. Selection uses |cross| separately.
_DRIFT_LINE_ORDER = (
    "base_error",
    "auto_drift",
    "error_drift_cross",
    "error_drift_total",
)
_DRIFT_LINE_COLORS = {
    "base_error": "tab:blue",
    "auto_drift": "tab:orange",
    "error_drift_cross": "tab:green",
    "error_drift_total": "tab:red",
}
_DRIFT_PANEL_LABELS = {
    "base_error": "base_error",
    "auto_drift": "auto_drift",
    "error_drift_cross": "error_drift_cross",
    "error_drift_total": "error_drift_total",
}


def val_criteria_plot_series(
    criteria,
    val_loss,
    val_spectrum,
    val_criteria_score,
    include_spectrum=None,
    drift=None,
):
    """Raw recon, spectrum, then the selection score.

    Checkpoint selection does not decide the panels. ``recon`` is always
    the raw validation loss. ``spectrum`` is the raw free-run spectral
    error, included when ``include_spectrum`` is true. The default is true
    when ``val_spectrum`` is an array, so passing ``None`` does not invent
    a zero series. The Appendix B terms are drawn by the overlay and the
    separate drift figure, not as extra rows here. ``drift`` is accepted
    so older callers keep working and is not turned into panels. The
    selection score is included only when more than one criterion is
    selected. Each entry is ``(name, values, selected)``. Arrays are used
    as given (already sliced to the current epoch).
    """
    del drift
    names = tuple(criteria)
    if include_spectrum is None:
        include_spectrum = val_spectrum is not None
    series = [
        ("recon", np.asarray(val_loss, dtype=float), "recon" in names),
    ]
    if include_spectrum:
        if val_spectrum is None:
            raise ValueError(
                "spectrum was requested but val_spectrum is missing."
            )
        series.append(
            (
                "spectrum",
                np.asarray(val_spectrum, dtype=float),
                "spectrum" in names,
            )
        )
    if len(names) > 1 and val_criteria_score is not None:
        series.append(
            ("combined", np.asarray(val_criteria_score, dtype=float), True)
        )
    return series


def _drift_arrays(drift, normalized=False):
    """The four Appendix B series, raw or divided by the variance of ``x_true``.

    ``normalized`` reads ``<name>_varnorm``. A missing variance-normalised
    series is left out rather than drawn as the raw values. None when no
    requested series is present.
    """
    if not drift:
        return None
    out = {}
    for name in _DRIFT_LINE_ORDER:
        key = name + "_varnorm" if normalized else name
        if key not in drift or drift[key] is None:
            continue
        out[name] = np.asarray(drift[key], dtype=float).reshape(-1)
    return out or None


def _has_varnorm(drift) -> bool:
    return _drift_arrays(drift, normalized=True) is not None


def _finite_limits(arrays, pad=0.05):
    chunks = []
    for arr in arrays:
        values = np.asarray(arr, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            chunks.append(values)
    if not chunks:
        return None
    vals = np.concatenate(chunks)
    lo = float(vals.min())
    hi = float(vals.max())
    if lo == hi:
        span = 1.0 if lo == 0.0 else abs(lo) * 0.05
        return lo - span, hi + span
    span = hi - lo
    return lo - pad * span, hi + pad * span


# 10th percentile of |cross|. A smaller percentile would follow a single
# near-zero sample and crush the rest of the axis into the log region.
_CROSS_LINTHRESH_PERCENTILE = 10.0
_CROSS_LINTHRESH_FLOOR = 1e-4


def cross_symlog_linthresh(cross) -> float:
    """Linear half-width for the signed cross term on a symlog axis.

    ``linthresh`` is the 10th percentile of the finite ``|cross|`` values,
    and at least 1e-4. A percentile of zero uses that floor. The plotted
    values stay signed. There is no absolute-value series and no sign marker.
    """
    values = np.abs(np.asarray(cross, dtype=float).reshape(-1))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return _CROSS_LINTHRESH_FLOOR
    thresh = float(np.percentile(values, _CROSS_LINTHRESH_PERCENTILE))
    if not np.isfinite(thresh) or thresh <= 0.0:
        return _CROSS_LINTHRESH_FLOOR
    return float(max(thresh, _CROSS_LINTHRESH_FLOOR))


def _symlog_linthresh(arrays) -> float:
    """Fallback linear half-width when a panel has no cross series.

    One decade below the smallest nonzero magnitude, and at least 1e-4.
    The cross term uses ``cross_symlog_linthresh`` instead.
    """
    chunks = []
    for arr in arrays:
        values = np.abs(np.asarray(arr, dtype=float).reshape(-1))
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            chunks.append(values)
    if not chunks:
        return 1e-3
    return float(max(np.min(np.concatenate(chunks)) * 0.1, 1e-4))


def _apply_symlog(ax, arrays, linthresh=None):
    ax.set_yscale(
        "symlog",
        linthresh=_symlog_linthresh(arrays) if linthresh is None else float(linthresh),
    )


def _criteria_panel_title(model_name, name, selected, normalized=False):
    """Legend title. Selected metrics are marked; the score panel is not."""
    if name == "combined":
        return f"{model_name}: selection score"
    label = _DRIFT_PANEL_LABELS.get(name)
    if label is None:
        label = "spectrum error" if name == "spectrum" else name
    if normalized:
        label = f"{label} / var"
    if selected:
        return f"{model_name}: {label} (selected)"
    return f"{model_name}: {label}"


def _mark_best_epoch(ax, values, best_epoch, labeled=True):
    """Black circle at the checkpoint epoch. Skipped when it is not drawable."""
    if best_epoch is None:
        return
    idx = int(best_epoch)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if idx < 0 or idx >= arr.size:
        return
    y = float(arr[idx])
    if not np.isfinite(y):
        return
    ax.plot(
        [idx],
        [y],
        linestyle="None",
        marker="o",
        color="black",
        markersize=8,
        label="best epoch" if labeled else None,
        zorder=5,
    )


def _style_loss_panel(ax, values, label, title, ylabel, best_epoch=None):
    ax.plot(values, label=label)
    _mark_best_epoch(ax, values, best_epoch)
    ax.legend(fontsize=16, title=title, title_fontsize=20)
    ax.set_xlabel("epochs", fontdict={"size": 16})
    ax.set_ylabel(ylabel, fontdict={"size": 16})
    ax.grid(True)


def _canon_criteria(criteria):
    from dvae.run_options import canonical_criterion_name

    names = []
    for name in criteria:
        canon = canonical_criterion_name(name)
        if canon and canon not in names:
            names.append(canon)
    return tuple(names)


def _criterion_selected(name, criteria, normalized=False) -> bool:
    """Raw panels follow the raw name. Varnorm panels follow ``name_varnorm``."""
    names = _canon_criteria(criteria)
    if normalized:
        return f"{name}_varnorm" in names
    return name in names


def _selected_label(name, criteria, normalized=False):
    label = _DRIFT_PANEL_LABELS.get(name, name)
    if normalized:
        label = f"{label} / var"
    if _criterion_selected(name, criteria, normalized=normalized):
        return f"{label} (selected)"
    return label


def _drift_linthresh(terms) -> float:
    """Symlog band for a drift axes. The cross series sets it when present."""
    if terms and "error_drift_cross" in terms:
        return cross_symlog_linthresh(terms["error_drift_cross"])
    if not terms:
        return _symlog_linthresh([])
    return _symlog_linthresh(list(terms.values()))


def _draw_warmup(
    ax,
    model_name,
    auto_sampling_method,
    noise_sampling_method,
    kl_warm_epochs,
    auto_warm_epochs,
    noise_warm_epochs,
    sequence_len_epochs,
    subband_warm_epochs=None,
    decorr_warm_epochs=None,
):
    draw_loss_warmup_lines(
        ax,
        model_name,
        auto_sampling_method,
        noise_sampling_method,
        kl_warm_epochs,
        auto_warm_epochs,
        noise_warm_epochs,
        sequence_len_epochs,
        subband_warm_epochs,
        decorr_warm_epochs,
    )


def _draw_drift_overlay(ax, drift, recon_k, criteria, best_epoch, title, normalized=False):
    """Four drift terms on one axes. ``recon_k`` sits behind the total."""
    if recon_k is not None:
        if normalized:
            recon_label = "recon_k / var"
        elif "recon_k" in _canon_criteria(criteria):
            recon_label = "recon_k (selected)"
        else:
            recon_label = "recon_k"
        ax.plot(
            np.asarray(recon_k, dtype=float).reshape(-1),
            color="0.75",
            linewidth=6,
            solid_capstyle="round",
            label=recon_label,
            zorder=1,
        )
    has_total = "error_drift_total" in drift
    labeled_best = False
    for name in _DRIFT_LINE_ORDER:
        if name not in drift:
            continue
        values = drift[name]
        on_top = name == "error_drift_total"
        ax.plot(
            values,
            color=_DRIFT_LINE_COLORS[name],
            linewidth=2.0 if on_top else 1.5,
            label=_selected_label(name, criteria, normalized=normalized),
            zorder=4 if on_top else 3,
        )
        label_this = on_top or (not has_total and not labeled_best)
        _mark_best_epoch(ax, values, best_epoch, labeled=label_this)
        labeled_best = labeled_best or label_this
    ax.legend(fontsize=12, title=title, title_fontsize=16)
    ax.set_xlabel("epochs", fontdict={"size": 16})
    ax.set_ylabel("drift terms / var" if normalized else "drift terms", fontdict={"size": 16})
    ax.grid(True)


def _lyapunov_line_label(text, criterion, criteria) -> str:
    if criterion in _canon_criteria(criteria):
        return f"{text} (selected)"
    return text


def _draw_lyapunov(ax, lyapunov, best_epoch, title, criteria=()):
    """One axes: finite-time slope, Rosenstein mean ± across-window std, GT line.

    The std band is the spread of the per-window Rosenstein estimates.
    Both curves and the ground-truth reference share this subplot.
    ``ftle`` and ``rosenstein_lle`` are marked when they are selected.
    """
    ftle = lyapunov.get("ftle")
    lle_mean = lyapunov.get("lle_mean")
    lle_std = lyapunov.get("lle_std")
    lle_gt = lyapunov.get("lle_gt")
    labeled_best = False
    if ftle is not None:
        ftle = np.asarray(ftle, dtype=float).reshape(-1)
        ax.plot(
            ftle,
            color="tab:purple",
            label=_lyapunov_line_label("finite-time exponent", "ftle", criteria),
        )
        _mark_best_epoch(ax, ftle, best_epoch, labeled=True)
        labeled_best = True
    if lle_mean is not None:
        mean = np.asarray(lle_mean, dtype=float).reshape(-1)
        if np.isfinite(mean).any():
            ax.plot(
                mean,
                color="tab:brown",
                label=_lyapunov_line_label(
                    "Rosenstein LLE", "rosenstein_lle", criteria
                ),
            )
            if lle_std is not None:
                std = np.asarray(lle_std, dtype=float).reshape(-1)
                n = int(min(mean.size, std.size))
                band = np.isfinite(mean[:n]) & np.isfinite(std[:n])
                if band.any():
                    xs = np.arange(n)
                    low = mean[:n] - std[:n]
                    high = mean[:n] + std[:n]
                    ax.fill_between(
                        xs[band],
                        low[band],
                        high[band],
                        color="tab:brown",
                        alpha=0.2,
                        label="Rosenstein ± std",
                    )
            _mark_best_epoch(ax, mean, best_epoch, labeled=not labeled_best)
    if lle_gt is not None and np.isfinite(float(lle_gt)):
        ax.axhline(
            float(lle_gt),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="GT LLE",
        )
    ax.legend(fontsize=12, title=title, title_fontsize=16)
    ax.set_xlabel("epochs", fontdict={"size": 16})
    ax.set_ylabel("exponent", fontdict={"size": 16})
    ax.grid(True)


def _lyapunov_present(lyapunov) -> bool:
    if not lyapunov:
        return False
    ftle = lyapunov.get("ftle")
    mean = lyapunov.get("lle_mean")
    if ftle is not None and np.asarray(ftle).size:
        return True
    if mean is not None and np.isfinite(np.asarray(mean, dtype=float)).any():
        return True
    return False


def build_val_criteria_figure(
    criteria,
    val_loss,
    val_spectrum,
    val_criteria_score,
    kl_warm_epochs,
    auto_warm_epochs,
    noise_warm_epochs,
    sequence_len_epochs,
    model_name,
    auto_sampling_method,
    noise_sampling_method,
    best_epoch=None,
    include_spectrum=None,
    drift=None,
    recon_k=None,
    lyapunov=None,
    log_scale=False,
    normalized=False,
    subband_warm_epochs=None,
    decorr_warm_epochs=None,
):
    """Recon, spectrum, the four drift terms overlaid, then Lyapunov.

    The overlay sits under the spectrum panel. ``recon_k`` is a thick
    light line behind ``error_drift_total``. Lyapunov is the finite-time
    exponent and, when it was computed, Rosenstein mean ± std with the
    ground-truth exponent as a horizontal line. The selection score is
    last, and only when more than one criterion is selected.

    ``log_scale`` uses a symlog y axis. On the drift overlay the linear
    band is the 10th percentile of finite ``|error_drift_cross|`` (floor
    1e-4), so a negative cross keeps its sign. There is no absolute-value
    series and no sign marker. ``normalized`` draws
    the four drift terms and ``recon_k`` divided by the variance of
    ``x_true``; recon, spectrum, and Lyapunov stay in their own units.
    Warm-up lines and the best-epoch marker are on every panel.
    """
    criteria = _canon_criteria(criteria)
    series = val_criteria_plot_series(
        criteria,
        val_loss,
        val_spectrum,
        val_criteria_score,
        include_spectrum=include_spectrum,
    )
    combined = None
    if series and series[-1][0] == "combined":
        combined = series.pop()
    drift_terms = _drift_arrays(drift, normalized=normalized)
    if normalized:
        var_total = None if drift is None else drift.get("error_drift_total_varnorm")
        recon_k = None if var_total is None else var_total
    blocks = [("series", item) for item in series]
    if drift_terms is not None:
        insert_at = 0
        for i, block in enumerate(blocks):
            if block[0] == "series" and block[1][0] == "spectrum":
                insert_at = i + 1
                break
        else:
            insert_at = len(blocks)
        blocks.insert(insert_at, ("overlay", drift_terms))
        if _lyapunov_present(lyapunov):
            blocks.insert(insert_at + 1, ("lyapunov", lyapunov))
    if combined is not None:
        blocks.append(("series", combined))
    n_panels = max(len(blocks), 1)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(8, 6 * n_panels), squeeze=False
    )
    plt.rcParams["font.size"] = 12
    ylabel = {
        "recon": "loss",
        "spectrum": "spectrum error",
        "combined": "normalised sum",
    }
    sym_arrays = []
    if drift_terms is not None:
        sym_arrays.extend(drift_terms.values())
    if recon_k is not None:
        sym_arrays.append(np.asarray(recon_k, dtype=float))
    linthresh = _drift_linthresh(drift_terms) if sym_arrays else None
    for ax, (kind, payload) in zip(axes[:, 0], blocks):
        _draw_warmup(
            ax,
            model_name,
            auto_sampling_method,
            noise_sampling_method,
            kl_warm_epochs,
            auto_warm_epochs,
            noise_warm_epochs,
            sequence_len_epochs,
            subband_warm_epochs,
            decorr_warm_epochs,
        )
        if kind == "overlay":
            _draw_drift_overlay(
                ax,
                payload,
                recon_k,
                criteria,
                best_epoch,
                (
                    f"{model_name}: drift terms / var"
                    if normalized
                    else f"{model_name}: drift terms"
                ),
                normalized=normalized,
            )
            if log_scale:
                _apply_symlog(ax, list(payload.values()) + (
                    [] if recon_k is None else [np.asarray(recon_k, dtype=float)]
                ), linthresh=linthresh)
        elif kind == "lyapunov":
            _draw_lyapunov(
                ax,
                payload,
                best_epoch,
                f"{model_name}: Lyapunov",
                criteria=criteria,
            )
            if log_scale:
                pieces = []
                for key in ("ftle", "lle_mean"):
                    if payload.get(key) is not None:
                        pieces.append(payload[key])
                if payload.get("lle_gt") is not None:
                    pieces.append([payload["lle_gt"]])
                _apply_symlog(ax, pieces or [[1.0]])
        else:
            name, values, selected = payload
            _style_loss_panel(
                ax,
                values,
                "validation",
                _criteria_panel_title(model_name, name, selected),
                ylabel.get(name, name),
                best_epoch=best_epoch,
            )
            if log_scale:
                _apply_symlog(ax, [values])
    if n_panels > 1:
        fig.tight_layout()
    return fig


def build_drift_panels_figure(
    drift,
    kl_warm_epochs,
    auto_warm_epochs,
    noise_warm_epochs,
    sequence_len_epochs,
    model_name,
    auto_sampling_method,
    noise_sampling_method,
    criteria=(),
    best_epoch=None,
    log_scale=False,
    normalized=False,
    subband_warm_epochs=None,
    decorr_warm_epochs=None,
):
    """One subplot per Appendix B term, with one shared y range.

    Order is base_error, auto_drift, error_drift_cross, error_drift_total.
    ``log_scale`` is a symlog axis with that same range. ``linthresh`` is
    the 10th percentile of the finite ``|cross|`` on these panels (floor
    1e-4), so a negative cross stays signed. ``normalized`` uses the
    ``*_varnorm`` series and marks a term selected only when that
    ``*_varnorm`` name is in ``criteria``. Warm-up lines and the
    best-epoch marker are on every subplot.
    """
    criteria = _canon_criteria(criteria)
    terms = _drift_arrays(drift, normalized=normalized)
    if terms is None:
        raise ValueError("drift panels need at least one Appendix B series.")
    names = [name for name in _DRIFT_LINE_ORDER if name in terms]
    limits = _finite_limits([terms[name] for name in names])
    linthresh = _drift_linthresh(terms)
    fig, axes = plt.subplots(
        len(names), 1, figsize=(8, 6 * len(names)), squeeze=False, sharey=True
    )
    plt.rcParams["font.size"] = 12
    for ax, name in zip(axes[:, 0], names):
        _draw_warmup(
            ax,
            model_name,
            auto_sampling_method,
            noise_sampling_method,
            kl_warm_epochs,
            auto_warm_epochs,
            noise_warm_epochs,
            sequence_len_epochs,
            subband_warm_epochs,
            decorr_warm_epochs,
        )
        selected = _criterion_selected(name, criteria, normalized=normalized)
        ylabel = _DRIFT_PANEL_LABELS[name]
        if normalized:
            ylabel = f"{ylabel} / var"
        _style_loss_panel(
            ax,
            terms[name],
            "validation",
            _criteria_panel_title(
                model_name, name, selected, normalized=normalized
            ),
            ylabel,
            best_epoch=best_epoch,
        )
        if log_scale:
            _apply_symlog(ax, [terms[name] for name in names], linthresh=linthresh)
        if limits is not None:
            ax.set_ylim(*limits)
    if len(names) > 1:
        fig.tight_layout()
    return fig


def _save_val_figure(fig, save_figures_dir, filename):
    fig_file = os.path.join(save_figures_dir, filename)
    fig.savefig(fig_file)
    plt.close(fig)
    return fig_file


def visualize_val_criteria(
    criteria,
    val_loss,
    val_spectrum,
    val_criteria_score,
    kl_warm_epochs,
    auto_warm_epochs,
    noise_warm_epochs,
    sequence_len_epochs,
    model_name,
    auto_sampling_method,
    noise_sampling_method,
    save_figures_dir,
    tag,
    best_epoch=None,
    include_spectrum=None,
    drift=None,
    recon_k=None,
    lyapunov=None,
    drift_norm="both",
    subband_warm_epochs=None,
    decorr_warm_epochs=None,
):
    """Save both validation layouts, each raw and variance-normalised.

    Each of those is saved on a linear y axis and a symlog y axis.
    ``drift_norm`` is ``raw``, ``varnorm``, or ``both`` (the default).
    The variance-normalised files use a ``_varnorm`` suffix, before
    ``_log`` when both apply. Log axes are symlog, so a negative
    ``error_drift_cross`` keeps its sign. Lyapunov is one subplot: the
    finite-time slope, the Rosenstein mean with its across-window std,
    and the ground-truth line.
    """
    plt.clf()
    common = dict(
        criteria=criteria,
        val_loss=val_loss,
        val_spectrum=val_spectrum,
        val_criteria_score=val_criteria_score,
        kl_warm_epochs=kl_warm_epochs,
        auto_warm_epochs=auto_warm_epochs,
        noise_warm_epochs=noise_warm_epochs,
        sequence_len_epochs=sequence_len_epochs,
        model_name=model_name,
        auto_sampling_method=auto_sampling_method,
        noise_sampling_method=noise_sampling_method,
        best_epoch=best_epoch,
        include_spectrum=include_spectrum,
        drift=drift,
        recon_k=recon_k,
        lyapunov=lyapunov,
        subband_warm_epochs=subband_warm_epochs,
        decorr_warm_epochs=decorr_warm_epochs,
    )
    modes = []
    if drift_norm in ("raw", "both"):
        modes.append((False, ""))
    if drift_norm in ("varnorm", "both") and _has_varnorm(drift):
        modes.append((True, "_varnorm"))
    if not modes:
        modes.append((False, ""))
    fig_file = None
    for normalized, norm_suffix in modes:
        for log_scale, log_suffix in ((False, ""), (True, "_log")):
            fig = build_val_criteria_figure(
                **common, log_scale=log_scale, normalized=normalized
            )
            saved = _save_val_figure(
                fig,
                save_figures_dir,
                f"vis_training_val_criteria_{tag}{norm_suffix}{log_suffix}.png",
            )
            if fig_file is None:
                fig_file = saved
    if _drift_arrays(drift) is not None:
        panel_kwargs = dict(
            drift=drift,
            kl_warm_epochs=kl_warm_epochs,
            auto_warm_epochs=auto_warm_epochs,
            noise_warm_epochs=noise_warm_epochs,
            sequence_len_epochs=sequence_len_epochs,
            model_name=model_name,
            auto_sampling_method=auto_sampling_method,
            noise_sampling_method=noise_sampling_method,
            criteria=criteria,
            best_epoch=best_epoch,
            subband_warm_epochs=subband_warm_epochs,
            decorr_warm_epochs=decorr_warm_epochs,
        )
        for normalized, norm_suffix in modes:
            if _drift_arrays(drift, normalized=normalized) is None:
                continue
            for log_scale, log_suffix in ((False, ""), (True, "_log")):
                fig = build_drift_panels_figure(
                    **panel_kwargs, log_scale=log_scale, normalized=normalized
                )
                _save_val_figure(
                    fig,
                    save_figures_dir,
                    f"vis_training_val_drift_{tag}{norm_suffix}{log_suffix}.png",
                )
    return fig_file


def visualize_total_loss(
    train_loss,
    val_loss,
    kl_warm_epochs,
    auto_warm_epochs,
    noise_warm_epochs,
    sequence_len_epochs,
    model_name,
    auto_sampling_method,
    noise_sampling_method,
    save_figures_dir,
    tag,
    log_scale=False,
    subband_warm_epochs=None,
    decorr_warm_epochs=None,
):
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 12
    draw_loss_warmup_lines(
        plt.gca(),
        model_name,
        auto_sampling_method,
        noise_sampling_method,
        kl_warm_epochs,
        auto_warm_epochs,
        noise_warm_epochs,
        sequence_len_epochs,
        subband_warm_epochs,
        decorr_warm_epochs,
    )
    plt.plot(train_loss, label="training loss")
    plt.plot(val_loss, label="validation loss")
    plt.legend(fontsize=16, title=f"{model_name}: Total Loss", title_fontsize=20)
    plt.xlabel("epochs", fontdict={"size": 16})
    plt.ylabel("loss", fontdict={"size": 16})
    if log_scale:
        plt.yscale("log")
        tag += "_log"
    plt.grid(True)
    fig_file = os.path.join(save_figures_dir, f"vis_training_loss_{tag}.png")
    plt.savefig(fig_file)
    plt.close(fig)


def visualize_recon_loss(
    train_recon,
    val_recon,
    kl_warm_epochs,
    auto_warm_epochs,
    noise_warm_epochs,
    sequence_len_epochs,
    model_name,
    auto_sampling_method,
    noise_sampling_method,
    save_figures_dir,
    tag,
    log_scale=False,
    subband_warm_epochs=None,
    decorr_warm_epochs=None,
):
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 12
    if model_name in ["VRNN", "MT_VRNN"]:
        for kl_warm_epoch in kl_warm_epochs:
            plt.axvline(x=kl_warm_epoch, color="c", linestyle="--")
    if auto_sampling_method in ["ss", "sm"]:
        for auto_warm_epoch in auto_warm_epochs:
            plt.axvline(x=auto_warm_epoch, color="r", linestyle=":")
    if noise_sampling_method in ["ss", "sm"]:
        for noise_warm_epoch in noise_warm_epochs:
            plt.axvline(x=noise_warm_epoch, color="g", linestyle="-.")
    for sequence_len_epoch in sequence_len_epochs:
        plt.axvline(x=sequence_len_epoch, color="m", linestyle="-.")
    draw_regularizer_warmup_lines(
        plt.gca(), subband_warm_epochs, decorr_warm_epochs
    )
    plt.plot(train_recon, label="Training")
    plt.plot(val_recon, label="Validation")
    plt.legend(fontsize=16, title=f"{model_name}: Recon. Loss", title_fontsize=20)
    plt.xlabel("epochs", fontdict={"size": 16})
    plt.ylabel("loss", fontdict={"size": 16})
    if log_scale:
        plt.yscale("log")
        tag += "_log"
    plt.grid(True)
    fig_file = os.path.join(save_figures_dir, f"vis_training_loss_recon_{tag}.png")
    plt.savefig(fig_file)
    plt.close(fig)


def visualize_kld_loss(
    train_kl,
    val_kl,
    kl_warm_epochs,
    auto_warm_epochs,
    noise_warm_epochs,
    sequence_len_epochs,
    model_name,
    auto_sampling_method,
    noise_sampling_method,
    save_figures_dir,
    tag,
    log_scale=False,
    subband_warm_epochs=None,
    decorr_warm_epochs=None,
):
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 12
    if model_name in ["VRNN", "MT_VRNN"]:
        for kl_warm_epoch in kl_warm_epochs:
            plt.axvline(x=kl_warm_epoch, color="c", linestyle="--")
    if auto_sampling_method in ["ss", "sm"]:
        for auto_warm_epoch in auto_warm_epochs:
            plt.axvline(x=auto_warm_epoch, color="r", linestyle=":")
    if noise_sampling_method in ["ss", "sm"]:
        for noise_warm_epoch in noise_warm_epochs:
            plt.axvline(x=noise_warm_epoch, color="g", linestyle="-.")
    for sequence_len_epoch in sequence_len_epochs:
        plt.axvline(x=sequence_len_epoch, color="m", linestyle="-.")
    draw_regularizer_warmup_lines(
        plt.gca(), subband_warm_epochs, decorr_warm_epochs
    )
    plt.plot(train_kl, label="Training")
    plt.plot(val_kl, label="Validation")
    plt.legend(fontsize=16, title=f"{model_name}: KL Div. Loss", title_fontsize=20)
    plt.xlabel("epochs", fontdict={"size": 16})
    plt.ylabel("loss", fontdict={"size": 16})
    if log_scale:
        plt.yscale("log")
        tag += "_log"
    plt.grid(True)
    fig_file = os.path.join(save_figures_dir, f"vis_training_loss_KLD_{tag}.png")
    plt.savefig(fig_file)
    plt.close(fig)


def _align_epoch_series(values, n):
    """Length-``n`` float series. Missing input is zeros."""
    if values is None:
        return np.zeros(n, dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == n:
        return arr
    out = np.full(n, np.nan, dtype=float)
    m = min(n, arr.size)
    if m:
        out[:m] = arr[:m]
    return out


def _format_window_stat(stats, key, state_key=None):
    """``off`` / ``na`` / a short float for the windowing text box."""
    if not stats:
        return "na"
    if state_key is not None and stats.get(state_key) == "off":
        return "off"
    value = stats.get(key)
    if value is None:
        return "na"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "na"
    if not np.isfinite(number):
        return "na"
    return f"{number:.6g}"


def _windowing_reference_text(dataset_stats):
    lines = [
        (
            "dataset missing rate",
            _format_window_stat(dataset_stats, "dataset_missing_rate"),
        ),
        (
            "NaN-ratio threshold",
            _format_window_stat(
                dataset_stats, "nan_ratio_threshold", "nan_ratio_threshold_state"
            ),
        ),
        (
            "p99 gap length",
            _format_window_stat(dataset_stats, "gap_quantile_len"),
        ),
        (
            "gap split threshold",
            _format_window_stat(
                dataset_stats, "gap_split_threshold", "gap_split_threshold_state"
            ),
        ),
        (
            "block mask fraction",
            _format_window_stat(dataset_stats, "block_mask_fraction"),
        ),
    ]
    return "\n".join(f"{name}: {value}" for name, value in lines)


def visualize_combined_metrics(
    delta_per_epoch,
    delta_threshold,
    kl_warm_epochs,
    kl_warm_values,
    auto_warm_epochs,
    auto_warm_values,
    noise_warm_epochs,
    noise_warm_values,
    sequence_len_epochs,
    sequence_len_values,
    cpt_patience_epochs,
    best_state_epochs,
    model_name,
    auto_sampling_method,
    noise_sampling_method,
    save_figures_dir,
    tag,
    patience_threshold_epochs=None,
    windows_kept=None,
    windows_dropped_nan=None,
    windows_dropped_shift=None,
    gap_splits=None,
    mean_start_shift=None,
    dataset_stats=None,
    timing=None,
    subband_warm_values=None,
    subband_loss_weight_values=None,
    decorr_warm_values=None,
    decorr_loss_weight_values=None,
    subband_warm_epochs=None,
    decorr_warm_epochs=None,
):
    """Warm-up, patience, windowing, and, when stored, epoch wall time.

    ``timing`` is the per-epoch seconds dict. Absent on an older history,
    which keeps the seven panels. A present dict adds a stacked wall-time
    panel. The same warm-up vertical lines are drawn on it.
    """
    plt.clf()
    n_panels = 8 if timing else 7
    fig, axs = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels))
    plt.rcParams["font.size"] = 12
    n_epochs = len(np.asarray(delta_per_epoch).reshape(-1))

    # Plot delta
    delta_above_threshold = np.where(
        delta_per_epoch >= delta_threshold, delta_per_epoch, np.nan
    )
    delta_below_threshold = np.where(
        delta_per_epoch < delta_threshold, delta_per_epoch, np.nan
    )
    axs[0].plot(delta_below_threshold, label=f"delta < {delta_threshold}", color="blue")
    axs[0].plot(
        delta_above_threshold, label=f"delta >= {delta_threshold}", color="orange"
    )
    axs[0].legend(fontsize=16, title="delta", title_fontsize=20, loc="upper left")
    axs[0].set_xlabel("epochs", fontdict={"size": 16})
    axs[0].set_ylabel("delta", fontdict={"size": 16})
    axs[0].grid(True)
    axs[0].set_yscale("symlog")

    if model_name in ["VRNN", "MT_VRNN"]:
        axs[1].step(
            np.arange(len(kl_warm_values)), kl_warm_values, label="kl_warm", color="c"
        )
    if auto_sampling_method in ["ss", "sm"]:
        axs[1].step(
            np.arange(len(auto_warm_values)),
            auto_warm_values,
            label="auto_warm",
            color="r",
        )

    if noise_sampling_method in ["ss", "sm"]:
        axs[1].step(
            np.arange(len(noise_warm_values)),
            noise_warm_values,
            label="noise_warm",
            color="g",
        )
    # Scale is the 0–1 warm-up progress. Weight is that scale times the
    # configured loss weight. None hides a disabled term.
    _regularizer_warm_steps = (
        (subband_warm_values, "subband scale", "tab:orange", "-"),
        (subband_loss_weight_values, "subband weight", "tab:orange", "--"),
        (decorr_warm_values, "decorr scale", "tab:purple", "-"),
        (decorr_loss_weight_values, "decorr weight", "tab:purple", "--"),
    )
    for values, label, color, linestyle in _regularizer_warm_steps:
        if values is None:
            continue
        series = np.asarray(values, dtype=float).reshape(-1)
        axs[1].step(
            np.arange(series.size),
            series,
            label=label,
            color=color,
            linestyle=linestyle,
        )

    if axs[1].get_legend_handles_labels()[1]:
        axs[1].legend(fontsize=16, title="warm values", title_fontsize=20)
    axs[1].set_xlabel("epochs", fontdict={"size": 16})
    axs[1].set_ylabel("warm values", fontdict={"size": 16})
    axs[1].grid(True)

    axs[2].step(
        np.arange(len(sequence_len_values)),
        sequence_len_values,
        label="sequence_len",
        color="m",
    )
    axs[2].legend(fontsize=16, title="sequence_len", title_fontsize=20)
    axs[2].set_xlabel("epochs", fontdict={"size": 16})
    axs[2].set_ylabel("sequence_len", fontdict={"size": 16})
    axs[2].grid(True)

    axs[3].step(
        range(len(cpt_patience_epochs)), cpt_patience_epochs, label="cpt_patience"
    )
    # Dashed threshold: the patience that would fire a bump or the final stop
    # on that epoch. Held at the warm-up start during sequence growth, then
    # rising over autonomous and KL bumps. Flat when the run uses a single
    # early_stop_patience.
    if patience_threshold_epochs is not None:
        axs[3].step(
            range(len(patience_threshold_epochs)),
            patience_threshold_epochs,
            linestyle="--",
            color="black",
            label="patience threshold",
        )
    axs[3].legend(fontsize=16, title="cpt_patience", title_fontsize=20)
    axs[3].set_xlabel("epochs", fontdict={"size": 16})
    axs[3].set_ylabel("cpt_patience", fontdict={"size": 16})
    axs[3].grid(True)

    axs[4].step(range(len(best_state_epochs)), best_state_epochs, label="best_state")
    axs[4].legend(fontsize=16, title="best_state", title_fontsize=20)
    axs[4].set_xlabel("epochs", fontdict={"size": 16})
    axs[4].set_ylabel("best_state", fontdict={"size": 16})
    axs[4].grid(True)

    # Windowing counts from each re-cut, held as steps between bumps.
    # Same step style as sequence_len. The text box is the one-time
    # dataset reference (missing rate, NaN threshold, p99 gap, split
    # threshold, block-mask fraction).
    count_series = (
        ("windows kept", _align_epoch_series(windows_kept, n_epochs), "k"),
        ("dropped NaN", _align_epoch_series(windows_dropped_nan, n_epochs), "tab:orange"),
        (
            "dropped shift",
            _align_epoch_series(windows_dropped_shift, n_epochs),
            "tab:brown",
        ),
        ("gap splits", _align_epoch_series(gap_splits, n_epochs), "tab:blue"),
    )
    for label, values, color in count_series:
        axs[5].step(np.arange(n_epochs), values, label=label, color=color)
    axs[5].legend(fontsize=16, title="windows", title_fontsize=20, loc="upper left")
    axs[5].text(
        0.99,
        0.98,
        _windowing_reference_text(dataset_stats),
        transform=axs[5].transAxes,
        va="top",
        ha="right",
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axs[5].set_xlabel("epochs", fontdict={"size": 16})
    axs[5].set_ylabel("windows", fontdict={"size": 16})
    axs[5].grid(True)

    axs[6].step(
        np.arange(n_epochs),
        _align_epoch_series(mean_start_shift, n_epochs),
        label="mean start shift",
        color="tab:purple",
    )
    axs[6].legend(fontsize=16, title="start shift", title_fontsize=20)
    axs[6].set_xlabel("epochs", fontdict={"size": 16})
    axs[6].set_ylabel("samples", fontdict={"size": 16})
    axs[6].grid(True)

    if timing:
        from dvae.epoch_log import TIMING_PLOT_LABELS

        stacks = []
        labels = []
        for key, label in TIMING_PLOT_LABELS:
            if key not in timing or timing[key] is None:
                continue
            values = np.asarray(timing[key], dtype=float).reshape(-1)
            if values.size < n_epochs:
                padded = np.zeros(n_epochs, dtype=float)
                padded[: values.size] = values
                values = padded
            stacks.append(values[:n_epochs])
            labels.append(label)
        if stacks:
            axs[7].stackplot(np.arange(n_epochs), *stacks, labels=labels)
        axs[7].legend(fontsize=12, title="wall time", title_fontsize=16, loc="upper left")
        axs[7].set_xlabel("epochs", fontdict={"size": 16})
        axs[7].set_ylabel("seconds", fontdict={"size": 16})
        axs[7].grid(True)

    if model_name in ["VRNN", "MT_VRNN"]:
        for kl_warm_epoch in kl_warm_epochs:
            for i in range(len(axs)):
                axs[i].axvline(x=kl_warm_epoch, color="c", linestyle="--")

    if auto_sampling_method in ["ss", "sm"]:
        for auto_warm_epoch in auto_warm_epochs:
            for i in range(len(axs)):
                axs[i].axvline(x=auto_warm_epoch, color="r", linestyle=":")

    if noise_sampling_method in ["ss", "sm"]:
        for noise_warm_epoch in noise_warm_epochs:
            for i in range(len(axs)):
                axs[i].axvline(x=noise_warm_epoch, color="g", linestyle="-.")

    for sequence_len_epoch in sequence_len_epochs:
        for i in range(len(axs)):
            axs[i].axvline(x=sequence_len_epoch, color="m", linestyle="-.")

    for i in range(len(axs)):
        draw_regularizer_warmup_lines(
            axs[i], subband_warm_epochs, decorr_warm_epochs
        )

    fig_file = os.path.join(
        save_figures_dir, f"vis_training_delta_kl_cpt_best_state_{tag}.png"
    )
    plt.savefig(fig_file)
    plt.close(fig)


def mt_alpha_from_sigma(sigmas):
    """Map stored MT sigmas to the values drawn in the alpha-history figure.

    This is the historical natural sigmoid used by MT_RNN plots and logs.
    The model itself mixes hidden states with a base-10 sigmoid. Do not
    change this default: existing MT figures depend on it.
    """
    return 1 / (1 + np.exp(-sigmas))


def _history_figure_tag(tag, file_tag):
    return tag if file_tag is None else file_tag


def _draw_warmup_markers(
    model_name,
    kl_warm_epochs,
    auto_warm_epochs,
    noise_warm_epochs,
    sequence_len_epochs,
    subband_warm_epochs=None,
    decorr_warm_epochs=None,
):
    if model_name in ["VRNN", "MT_VRNN"] and kl_warm_epochs is not None:
        for kl_warm_epoch in kl_warm_epochs:
            plt.axvline(x=kl_warm_epoch, color="c", linestyle="--")
    if auto_warm_epochs is not None:
        for auto_warm_epoch in auto_warm_epochs:
            plt.axvline(x=auto_warm_epoch, color="r", linestyle=":")
    if noise_warm_epochs is not None:
        for noise_warm_epoch in noise_warm_epochs:
            plt.axvline(x=noise_warm_epoch, color="g", linestyle="-.")
    if sequence_len_epochs is not None:
        for sequence_len_epoch in sequence_len_epochs:
            plt.axvline(x=sequence_len_epoch, color="m", linestyle="-.")
    draw_regularizer_warmup_lines(
        plt.gca(), subband_warm_epochs, decorr_warm_epochs
    )


def visualize_sigma_history(
    sigmas_history,
    model_name,
    save_figures_dir,
    tag,
    kl_warm_epochs=None,
    auto_warm_epochs=None,
    noise_warm_epochs=None,
    sequence_len_epochs=None,
    *,
    file_tag=None,
    series_name="Sigma",
    legend_title="Sigma values",
    ylabel="sigma",
    subband_warm_epochs=None,
    decorr_warm_epochs=None,
):
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 12
    _draw_warmup_markers(
        model_name,
        kl_warm_epochs,
        auto_warm_epochs,
        noise_warm_epochs,
        sequence_len_epochs,
        subband_warm_epochs,
        decorr_warm_epochs,
    )

    for i in range(sigmas_history.shape[0]):
        plt.plot(sigmas_history[i], label=f"{series_name} {i+1}")
    plt.legend(fontsize=16, title=legend_title, title_fontsize=20)
    plt.xlabel("epochs", fontdict={"size": 16})
    plt.ylabel(ylabel, fontdict={"size": 16})
    plt.grid(True)
    fig_file = os.path.join(
        save_figures_dir,
        f"vis_training_history_of_sigma_{_history_figure_tag(tag, file_tag)}.png",
    )
    plt.savefig(fig_file)
    plt.close(fig)


def visualize_alpha_history(
    sigmas_history,
    model_name,
    save_figures_dir,
    tag,
    kl_warm_epochs=None,
    auto_warm_epochs=None,
    noise_warm_epochs=None,
    sequence_len_epochs=None,
    *,
    file_tag=None,
    alpha_from_sigma=None,
    series_name="Alpha",
    legend_title="Alpha values",
    ylabel="alpha",
    subband_warm_epochs=None,
    decorr_warm_epochs=None,
):
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 12
    _draw_warmup_markers(
        model_name,
        kl_warm_epochs,
        auto_warm_epochs,
        noise_warm_epochs,
        sequence_len_epochs,
        subband_warm_epochs,
        decorr_warm_epochs,
    )

    transform = mt_alpha_from_sigma if alpha_from_sigma is None else alpha_from_sigma
    for i in range(sigmas_history.shape[0]):
        alphas = transform(sigmas_history[i])
        plt.plot(alphas, label=f"{series_name} {i+1}")
    plt.legend(fontsize=16, title=legend_title, title_fontsize=20)
    plt.xlabel("epochs", fontdict={"size": 16})
    plt.ylabel(ylabel, fontdict={"size": 16})
    plt.grid(True)
    plt.yscale("log")
    plt.ylim(1e-2, 1e1)
    fig_file = os.path.join(
        save_figures_dir,
        f"vis_training_history_of_alpha_{_history_figure_tag(tag, file_tag)}.png",
    )
    plt.savefig(fig_file)
    plt.close(fig)


# One row per fixed window, one column per timestep of that window.
# The last flag fixes a linear cosine axis on [-1, 1].
_DRIFT_STEP_PANELS = (
    ("e2", r"$e_t^{2}$", "tab:blue", False),
    ("ed", r"$e_t d_t$", "tab:green", False),
    ("d2", r"$d_t^{2}$", "tab:orange", False),
    ("cos", r"$\cos_t$", "tab:red", True),
)


def _align_drift_timestep_axes(fig, axes, time):
    """Same x limits and ticks. Y labels share a left edge. Y scales stay per panel."""
    left = float(time[0])
    right = float(time[-1]) if time.size > 1 else left + 1.0
    axes[-1].set_xlim(left, right)
    fig.canvas.draw()
    xticks = np.asarray(axes[-1].get_xticks(), dtype=float)
    for ax in axes:
        ax.set_xlim(left, right)
        ax.set_xticks(xticks)
    fig.align_ylabels(axes)


def build_drift_timestep_figure(terms, flip=None, normalized=False):
    """Four stacked panels of per-window drift trajectories.

    ``terms`` maps ``e2``, ``ed``, ``d2``, and ``cos`` to arrays shaped
    ``(window, time)``. Each panel draws every window faintly and the
    mean across windows as a solid line. The time axis, its limits, and
    its ticks are shared. Each panel keeps its own y scale. The cosine
    panel is linear on ``[-1, 1]``. ``flip`` marks the teacher-forced
    boundary when it is known. Styling follows ``aggregate_plot_style``.
    """
    saved_style = plt.rcParams.copy()
    apply_paper_ready_line_style()
    try:
        fig, axes = plt.subplots(
            len(_DRIFT_STEP_PANELS), 1, sharex=True, figsize=(12, 18), squeeze=True
        )
        time = None
        for ax, (key, label, color, cosine_axis) in zip(axes, _DRIFT_STEP_PANELS):
            values = np.asarray(terms[key], dtype=float)
            if values.ndim != 2:
                raise ValueError(
                    f"drift timestep {key} must have shape (window, time), "
                    f"got {values.shape}."
                )
            if time is None:
                time = np.arange(values.shape[1])
            elif values.shape[1] != time.size:
                raise ValueError(
                    "drift timestep panels must share a time axis, "
                    f"got {time.size} and {values.shape[1]}."
                )
            for index in range(values.shape[0]):
                ax.plot(
                    time,
                    values[index],
                    color=color,
                    alpha=0.2,
                    linewidth=1.0,
                    label="window" if index == 0 else "_nolegend_",
                    zorder=1,
                )
            finite = np.isfinite(values)
            count = finite.sum(axis=0)
            summed = np.sum(np.where(finite, values, 0.0), axis=0)
            mean = np.full(count.shape, np.nan, dtype=float)
            ok = count > 0
            mean[ok] = summed[ok] / count[ok]
            ax.plot(
                time,
                mean,
                color=color,
                alpha=1.0,
                linestyle="-",
                label="mean",
                zorder=3,
            )
            if flip is not None:
                ax.axvline(
                    float(flip),
                    color="0.35",
                    linestyle="--",
                    linewidth=1.0,
                    label="_nolegend_",
                    zorder=2,
                )
            if cosine_axis:
                ylabel = label
            else:
                ylabel = label + (r" / var" if normalized else "")
            ax.set_ylabel(ylabel)
            if cosine_axis:
                ax.set_yscale("linear")
                ax.set_ylim(-1.0, 1.0)
                ax.set_autoscaley_on(False)
            else:
                setup_plot_y_axis(ax, values)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel(r"$t$")
        _align_drift_timestep_axes(fig, axes, time)
        return fig
    finally:
        plt.rcParams.update(saved_style)


def save_drift_timestep_figure(terms, path, flip=None, normalized=False):
    """Write the four-panel drift trajectory figure."""
    fig = build_drift_timestep_figure(terms, flip=flip, normalized=normalized)
    try:
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.align_ylabels(fig.axes)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    finally:
        plt.close(fig)


# Own folder under vis_during_training, same layout as
# sequence_reconstruction/half_half, even_bursts, and training_mode.
# The epoch is in the filename so each checkpoint keeps its own figure.
DRIFT_TIMESTEP_SUBDIR = "drift_timestep"


def drift_timestep_figure_path(save_figures_dir, epoch, normalized=False):
    """Path of the checkpoint drift-trajectory figure.

    Raw:
    ``{save_figures_dir}/drift_timestep/vis_drift_timestep_epoch_{epoch}.png``

    Variance-normalised:
    ``{save_figures_dir}/drift_timestep/vis_drift_timestep_varnorm_epoch_{epoch}.png``

    ``save_figures_dir`` is the run's ``vis_during_training`` directory.
    """
    stem = (
        "vis_drift_timestep_varnorm_epoch_{}.png"
        if normalized
        else "vis_drift_timestep_epoch_{}.png"
    )
    return os.path.join(
        save_figures_dir, DRIFT_TIMESTEP_SUBDIR, stem.format(int(epoch))
    )


def save_drift_timestep_figures(
    terms, save_figures_dir, epoch, flip=None, varnorm_terms=None
):
    """Draw the checkpoint drift trajectories into their own subdirectory.

    ``terms`` and ``varnorm_terms`` are the current epoch only. Nothing
    here is written into the training history. Returns the paths written.
    """
    path = drift_timestep_figure_path(save_figures_dir, epoch, normalized=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_drift_timestep_figure(terms, path, flip=flip, normalized=False)
    written = [path]
    if varnorm_terms is not None:
        var_path = drift_timestep_figure_path(
            save_figures_dir, epoch, normalized=True
        )
        save_drift_timestep_figure(
            varnorm_terms, var_path, flip=flip, normalized=True
        )
        written.append(var_path)
    return written
