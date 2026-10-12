import matplotlib.pyplot as plt
import os
import numpy as np


def draw_loss_warmup_lines(
    ax,
    model_name,
    auto_sampling_method,
    noise_sampling_method,
    kl_warm_epochs,
    auto_warm_epochs,
    noise_warm_epochs,
    sequence_len_epochs,
):
    """Vertical warmup markers used by the loss-over-epochs figure.

    Cyan dashed: KL warm-up, VRNN / MT_VRNN only. Red dotted: autonomous
    warm-up when the sampler is ss or sm. Green dash-dot: noise warm-up
    on the same samplers. Magenta dash-dot: sequence-length changes.
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


def val_criteria_plot_series(
    criteria,
    val_loss,
    val_spectrum,
    val_criteria_score,
    include_spectrum=None,
):
    """Raw recon, spectrum error when it was computed, then the selection score.

    Checkpoint selection does not decide the panels. ``recon`` is always
    the raw validation loss. ``spectrum`` is the raw free-run spectral
    error, included when ``include_spectrum`` is true. The default is true
    when ``val_spectrum`` is an array, so passing ``None`` does not invent
    a zero series. The selection score is included only when more than one
    criterion is selected. Each entry is ``(name, values, selected)``.
    Arrays are used as given (already sliced to the current epoch).
    """
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


def _criteria_panel_title(model_name, name, selected):
    """Legend title. Selected metrics are marked; the score panel is not."""
    if name == "combined":
        return f"{model_name}: selection score"
    label = "spectrum error" if name == "spectrum" else name
    if selected:
        return f"{model_name}: {label} (selected)"
    return f"{model_name}: {label}"


def _mark_best_epoch(ax, values, best_epoch):
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
        label="best epoch",
        zorder=5,
    )


def _style_loss_panel(ax, values, label, title, ylabel, best_epoch=None):
    ax.plot(values, label=label)
    _mark_best_epoch(ax, values, best_epoch)
    ax.legend(fontsize=16, title=title, title_fontsize=20)
    ax.set_xlabel("epochs", fontdict={"size": 16})
    ax.set_ylabel(ylabel, fontdict={"size": 16})
    ax.grid(True)


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
):
    """One axes per plotted metric, with warm-up lines and a best-epoch marker.

    Recon is always drawn. Spectrum is drawn when it was computed.
    The selection score is drawn when more than one criterion is selected.
    ``best_epoch`` is a point on each series, not another vertical line.
    """
    series = val_criteria_plot_series(
        criteria,
        val_loss,
        val_spectrum,
        val_criteria_score,
        include_spectrum=include_spectrum,
    )
    n_panels = max(len(series), 1)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(8, 6 * n_panels), squeeze=False
    )
    plt.rcParams["font.size"] = 12
    ylabel = {
        "recon": "loss",
        "spectrum": "spectrum error",
        "combined": "normalised sum",
    }
    for ax, (name, values, selected) in zip(axes[:, 0], series):
        draw_loss_warmup_lines(
            ax,
            model_name,
            auto_sampling_method,
            noise_sampling_method,
            kl_warm_epochs,
            auto_warm_epochs,
            noise_warm_epochs,
            sequence_len_epochs,
        )
        _style_loss_panel(
            ax,
            values,
            "validation",
            _criteria_panel_title(model_name, name, selected),
            ylabel.get(name, name),
            best_epoch=best_epoch,
        )
    if n_panels > 1:
        fig.tight_layout()
    return fig


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
):
    """Save validation criteria over epochs at the loss-figure cadence."""
    plt.clf()
    fig = build_val_criteria_figure(
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
        best_epoch=best_epoch,
        include_spectrum=include_spectrum,
    )
    fig_file = os.path.join(save_figures_dir, f"vis_training_val_criteria_{tag}.png")
    fig.savefig(fig_file)
    plt.close(fig)
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
):
    plt.clf()
    fig, axs = plt.subplots(7, 1, figsize=(12, 21))
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
