"""Training-time timescale traces for PLRNN-family cells.

MT_RNN / MT_VRNN already record ``model.sigmas`` and, on each
``save_frequency`` checkpoint, write

    vis_during_training/vis_training_history_of_sigma_{tag}.png
    vis_during_training/vis_training_history_of_alpha_{tag}.png

PLRNN and shPLRNN do not have those mixing sigmas. Their per-unit
timescale is the diagonal decay ``A = sigmoid_10(A_sigmas)`` on the
recurrent cell (``model.rnn.A_sigmas``). This module records that
parameter on the same cadence and feeds it through the same figure
functions.

MT plots still convert sigmas with a natural sigmoid. That path is
unchanged. PLRNN alpha curves use the cell's base-10 sigmoid so the
figure shows the diagonal ``A`` the recurrence actually applies.
"""

import numpy as np

MT_MODEL_NAMES = ("MT_RNN", "MT_VRNN")


def diagonal_A_from_sigmas(sigmas):
    """Base-10 sigmoid used by PLRNN / shPLRNN ``A`` (not MT's natural sigmoid)."""
    sigmas = np.asarray(sigmas, dtype=np.float64)
    return 1.0 / (1.0 + np.power(10.0, -sigmas))


def cell_A_sigmas(model):
    """Return the trainable diagonal-A logits, or None when the cell has none."""
    cell = getattr(model, "rnn", None)
    if cell is None:
        return None
    sigmas = getattr(cell, "A_sigmas", None)
    if sigmas is None or not hasattr(sigmas, "detach"):
        return None
    return sigmas


def _flat_A_sigmas(model):
    sigmas = cell_A_sigmas(model)
    if sigmas is None:
        return None
    return sigmas.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)


def init_A_sigmas_history(model, epochs):
    """Allocate ``(n_units, epochs)`` and store the current ``A_sigmas`` in column 0.

    Returns None for vanilla RNN / LSTM / shPLRNN_wo_A (no ``A_sigmas``).
    """
    flat = _flat_A_sigmas(model)
    if flat is None:
        return None
    history = np.zeros((flat.shape[0], int(epochs)), dtype=np.float64)
    if history.shape[1] > 0:
        history[:, 0] = flat
    return history


def update_A_sigmas_history(history, model, epoch):
    """Write the current ``A_sigmas`` into one epoch column. No-op without history."""
    if history is None:
        return
    flat = _flat_A_sigmas(model)
    if flat is None or flat.shape[0] != history.shape[0]:
        return
    if epoch < 0 or epoch >= history.shape[1]:
        return
    history[:, epoch] = flat


def plrnn_figure_tag(model_name, tag):
    """Filename tag for PLRNN timescale figures.

    Non-MT runs reuse the MT filename pattern (``..._{tag}.png``).
    MT runs that also contain a PLRNN cell get a ``_diagA`` suffix so
    those files do not replace the mixing-alpha figures.
    """
    if model_name in MT_MODEL_NAMES:
        return f"{tag}_diagA"
    return tag


def format_scalar_list(values):
    """Same ``['0.12345', ...]`` list MT uses in ``alphas:`` log lines."""
    return [f"{float(value):.5f}" for value in np.asarray(values).reshape(-1)]


def log_plrnn_timescales(logger, history, epoch, *, final=False, alongside_mt=False):
    """Log diagonal A and A_sigmas. No-op when the cell has no such history."""
    if history is None or logger is None:
        return
    sigmas = np.asarray(history[:, epoch], dtype=np.float64).reshape(-1)
    alphas = diagonal_A_from_sigmas(sigmas)
    prefix = "Final " if final else ""
    # Alongside MT, keep the existing ``alphas:`` line meaning mixing alphas.
    alpha_key = "diagonal A" if alongside_mt else "alphas"
    logger.info(
        "{}{}: {}".format(prefix, alpha_key, format_scalar_list(alphas))
    )
    logger.info("{}A_sigmas: {}".format(prefix, format_scalar_list(sigmas)))


def save_plrnn_timescale_figures(
    history,
    model_name,
    save_figures_dir,
    tag,
    kl_warm_epochs=None,
    auto_warm_epochs=None,
    noise_warm_epochs=None,
    sequence_len_epochs=None,
    *,
    visualize_sigma_history=None,
    visualize_alpha_history=None,
):
    """Write the same sigma/alpha history figures MT writes, for diagonal A.

    ``history`` rows are ``A_sigmas``. The sigma figure plots those logits.
    The alpha figure plots ``sigmoid_10(A_sigmas)``, i.e. diagonal A.
    Returns the filename tag, or None when there is nothing to plot.
    """
    if history is None:
        return None
    if visualize_sigma_history is None or visualize_alpha_history is None:
        from dvae.visualizers.visualize_training_metrics import (
            visualize_alpha_history as _alpha_plot,
            visualize_sigma_history as _sigma_plot,
        )

        if visualize_sigma_history is None:
            visualize_sigma_history = _sigma_plot
        if visualize_alpha_history is None:
            visualize_alpha_history = _alpha_plot

    file_tag = plrnn_figure_tag(model_name, tag)
    visualize_sigma_history(
        history,
        model_name,
        save_figures_dir,
        tag,
        kl_warm_epochs,
        auto_warm_epochs,
        noise_warm_epochs,
        sequence_len_epochs,
        file_tag=file_tag,
        series_name="A_sigma",
        legend_title="A_sigma values",
        ylabel="A_sigma",
    )
    visualize_alpha_history(
        history,
        model_name,
        save_figures_dir,
        tag,
        kl_warm_epochs,
        auto_warm_epochs,
        noise_warm_epochs,
        sequence_len_epochs,
        file_tag=file_tag,
        alpha_from_sigma=diagonal_A_from_sigmas,
        series_name="A",
        legend_title="diagonal A",
        ylabel="A",
    )
    return file_tag


def attach_timescale_history(pickle_dict, *, sigmas_history=None, a_sigmas_history=None):
    """Persist timescale traces in ``loss_model.pckl``.

    MT mixing sigmas keep the ``sigmas_history`` key. PLRNN logits are
    stored as ``A_sigmas_history``. When the run is not MT, they are also
    stored as ``sigmas_history`` so the history dump MT writes is present.
    """
    if sigmas_history is not None:
        pickle_dict["sigmas_history"] = sigmas_history
    if a_sigmas_history is not None:
        pickle_dict["A_sigmas_history"] = a_sigmas_history
        if "sigmas_history" not in pickle_dict:
            pickle_dict["sigmas_history"] = a_sigmas_history
    return pickle_dict
