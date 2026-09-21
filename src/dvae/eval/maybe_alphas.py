"""Safe access to per-unit timescales during evaluation."""


def maybe_alphas_per_unit(dvae):
    """Return per-unit timescales if the model exposes them; otherwise None.

    Sources (must be a real callable; no placeholders):
    - MT_RNN / MT_VRNN: mixing ``alphas_per_unit()``
    - BaseRNN with type_rnn in {PLRNN, shPLRNN}: constrained diagonal A
    - vanilla RNN / LSTM / models without the method: None
    """
    getter = getattr(dvae, "alphas_per_unit", None)
    if not callable(getter):
        return None
    return getter()


def alphas_to_metric_list(alphas):
    """YAML-safe list of floats, or None when the model has no timescales."""
    if alphas is None:
        return None
    if hasattr(alphas, "detach"):
        return [float(a) for a in alphas.detach().cpu().reshape(-1)]
    return [float(a) for a in alphas]
