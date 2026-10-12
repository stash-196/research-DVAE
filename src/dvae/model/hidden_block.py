"""Heart/brain split of the recurrent map and the decoder.

Blank, off, or false leaves the model unchanged. ``true``, ``on``, or
``half`` splits ``dim_rnn`` in half (an odd width gives the extra unit
to the brain block). ``a,b`` must sum to ``dim_rnn``.

The input matrix is not split. Off-block recurrent entries are zero
unless weak coupling is on, in which case they are multiplied by that
factor (default 0.01). Decoder cross terms stay hard zero: heart units
write only the configured ECG channels, brain units only the EEG
channels. PLRNN still zeros its diagonal by multiplying ``W_mask``;
the block mask is an extra factor, not a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

DEFAULT_WEAK_COUPLING = 0.01
_OFF = {"", "none", "null", "off", "false", "no", "n", "0"}
# Channel lists: blank, off, and none are unset. "0" is channel 0.
_CHANNEL_UNSET = {"", "off", "none"}
_HALF = {"true", "yes", "y", "on", "half", "1"}
_COUPLING_ON = {"true", "yes", "y", "on"}


@dataclass(frozen=True)
class HiddenBlockSplit:
    heart: int
    brain: int
    coupling: float
    ecg: tuple[int, ...]
    eeg: tuple[int, ...]
    heart_alphas: tuple[float, ...] | None = None
    brain_alphas: tuple[float, ...] | None = None
    heart_alpha_range: tuple[float, float] | None = None
    brain_alpha_range: tuple[float, float] | None = None

    @property
    def dim(self) -> int:
        return self.heart + self.brain


def _text(raw) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def _split_sizes(raw: str, dim_rnn: int) -> tuple[int, int]:
    text = raw.strip().lower()
    if text in _HALF:
        if dim_rnn < 2:
            raise ValueError(
                f"hidden_block_split needs dim_rnn >= 2, got {dim_rnn}."
            )
        heart = dim_rnn // 2
        return heart, dim_rnn - heart
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(
            "hidden_block_split must be off, half, or two sizes, "
            f"got {raw!r}."
        )
    try:
        heart, brain = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"hidden_block_split sizes must be integers, got {raw!r}."
        ) from exc
    if heart < 1 or brain < 1:
        raise ValueError(
            f"hidden_block_split sizes must both be >= 1, got {raw!r}."
        )
    if heart + brain != dim_rnn:
        raise ValueError(
            "hidden_block_split sizes must sum to dim_rnn "
            f"({dim_rnn}), got {heart}+{brain}."
        )
    return heart, brain


def _coupling(raw) -> float:
    text = _text(raw).lower()
    if text in _OFF:
        return 0.0
    if text in _COUPLING_ON:
        return DEFAULT_WEAK_COUPLING
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(
            "hidden_block_weak_coupling must be off, true, or a "
            f"non-negative float, got {raw!r}."
        ) from exc
    if not (value >= 0.0) or value > 1.0:
        raise ValueError(
            f"hidden_block_weak_coupling must be in [0, 1], got {value!r}."
        )
    return value


def _index_list(raw, name: str, n_channels: int) -> tuple[int, ...]:
    text = _text(raw)
    if text.lower() in _CHANNEL_UNSET:
        raise ValueError(
            f"{name} is required when hidden_block_split is on, got {raw!r}."
        )
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{name} is required when hidden_block_split is on.")
    try:
        indices = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{name} must be 0-based integers, got {raw!r}.") from exc
    if len(indices) != len(set(indices)):
        raise ValueError(f"{name} repeats an index, got {raw!r}.")
    for index in indices:
        if index < 0 or index >= n_channels:
            raise ValueError(
                f"{name} index {index} is outside 0..{n_channels - 1}."
            )
    return indices


def _alpha_list(raw, name: str) -> tuple[float, ...] | None:
    text = _text(raw)
    if text.lower() in _OFF:
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list, got {raw!r}.") from exc
    for value in values:
        if not (0.0 < value < 1.0):
            raise ValueError(f"{name} entries must be in (0, 1), got {value!r}.")
    return values


def _alpha_range(raw, name: str) -> tuple[float, float] | None:
    text = _text(raw)
    if text.lower() in _OFF:
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must be min,max, got {raw!r}.")
    try:
        low, high = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"{name} must be min,max, got {raw!r}.") from exc
    if not (0.0 < low < 1.0 and 0.0 < high < 1.0):
        raise ValueError(f"{name} endpoints must be in (0, 1), got {raw!r}.")
    if low > high:
        raise ValueError(f"{name} min must be <= max, got {raw!r}.")
    return low, high


def _option(cfg, option: str):
    if not cfg.has_section("Network"):
        return None
    if not cfg.has_option("Network", option):
        return None
    return cfg.get("Network", option)


def parse_hidden_block_split(cfg, dim_rnn: int, y_dim: int) -> HiddenBlockSplit | None:
    """``None`` when the split is blank or off."""
    raw = _option(cfg, "hidden_block_split")
    text = _text(raw).lower()
    if text in _OFF:
        return None
    heart, brain = _split_sizes(_text(raw), int(dim_rnn))
    ecg = _index_list(_option(cfg, "ecg_channels"), "ecg_channels", int(y_dim))
    eeg = _index_list(_option(cfg, "eeg_channels"), "eeg_channels", int(y_dim))
    overlap = set(ecg) & set(eeg)
    if overlap:
        raise ValueError(
            "ecg_channels and eeg_channels must be disjoint, "
            f"overlap {sorted(overlap)}."
        )
    heart_alphas = _alpha_list(_option(cfg, "heart_alphas"), "heart_alphas")
    brain_alphas = _alpha_list(_option(cfg, "brain_alphas"), "brain_alphas")
    heart_range = _alpha_range(_option(cfg, "heart_alpha_range"), "heart_alpha_range")
    brain_range = _alpha_range(_option(cfg, "brain_alpha_range"), "brain_alpha_range")
    if heart_alphas is not None and heart_range is not None:
        raise ValueError("Set heart_alphas or heart_alpha_range, not both.")
    if brain_alphas is not None and brain_range is not None:
        raise ValueError("Set brain_alphas or brain_alpha_range, not both.")
    return HiddenBlockSplit(
        heart=heart,
        brain=brain,
        coupling=_coupling(_option(cfg, "hidden_block_weak_coupling")),
        ecg=ecg,
        eeg=eeg,
        heart_alphas=heart_alphas,
        brain_alphas=brain_alphas,
        heart_alpha_range=heart_range,
        brain_alpha_range=brain_range,
    )


def recurrent_block_mask(heart: int, brain: int, coupling: float) -> torch.Tensor:
    """Square map. Within-block entries are 1. Off-block entries are ``coupling``."""
    width = heart + brain
    mask = torch.full((width, width), float(coupling))
    mask[:heart, :heart] = 1.0
    mask[heart:, heart:] = 1.0
    return mask


def shallow_split_sizes(heart: int, brain: int, hidden_sh: int) -> tuple[int, int]:
    """Proportional split of the shPLRNN shallow width. Both sides stay >= 1."""
    if hidden_sh < 2:
        raise ValueError(
            "shPLRNN hidden_block_split needs hidden_sh_size >= 2, "
            f"got {hidden_sh}."
        )
    heart_sh = int(round(hidden_sh * heart / (heart + brain)))
    heart_sh = min(max(heart_sh, 1), hidden_sh - 1)
    return heart_sh, hidden_sh - heart_sh


def shallow_factor_masks(
    heart: int, brain: int, heart_sh: int, brain_sh: int, coupling: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``W1`` is (dz, dh) and ``W2`` is (dh, dz). Cross terms use ``coupling``."""
    w1 = torch.full((heart + brain, heart_sh + brain_sh), float(coupling))
    w1[:heart, :heart_sh] = 1.0
    w1[heart:, heart_sh:] = 1.0
    w2 = torch.full((heart_sh + brain_sh, heart + brain), float(coupling))
    w2[:heart_sh, :heart] = 1.0
    w2[heart_sh:, heart:] = 1.0
    return w1, w2


def decoder_mask(
    heart: int, brain: int, y_dim: int, ecg: tuple[int, ...], eeg: tuple[int, ...]
) -> torch.Tensor:
    """``nn.Linear`` weight layout ``(y_dim, dim_rnn)``. Cross terms are 0."""
    mask = torch.zeros(y_dim, heart + brain)
    for index in ecg:
        mask[index, :heart] = 1.0
    for index in eeg:
        mask[index, heart:] = 1.0
    return mask


def _tile_tensor(src: torch.Tensor, n: int) -> torch.Tensor:
    """Repeat ``src`` along the block. The same tensor object is reused so grad accumulates."""
    src = src.reshape(-1)
    if n <= src.numel():
        return src[:n]
    repeats = n // src.numel()
    remainder = n % src.numel()
    parts = [src for _ in range(repeats)]
    if remainder:
        parts.append(src[:remainder])
    return torch.cat(parts)


def _tile_values(values: tuple[float, ...], n: int, like: torch.Tensor) -> torch.Tensor:
    src = torch.tensor(values, dtype=like.dtype, device=like.device)
    return _tile_tensor(src, n)


def _range_values(
    bounds: tuple[float, float], n: int, like: torch.Tensor
) -> torch.Tensor:
    return torch.linspace(
        bounds[0], bounds[1], n, dtype=like.dtype, device=like.device
    )


def block_alpha_init(values, bounds, width) -> torch.Tensor | None:
    """Alpha-space start for one block, or None when that block stays shared.

    A comma list is the learnable vector (tiled over the block in the
    forward). A min,max range is one sample per unit, spread with linspace.
    """
    if bounds is not None:
        low, high = bounds
        return torch.linspace(float(low), float(high), int(width))
    if values is not None:
        return torch.tensor(list(values), dtype=torch.float32)
    return None


def apply_trainable_block_alphas(
    shared: torch.Tensor,
    split: HiddenBlockSplit,
    heart: torch.Tensor | None = None,
    brain: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write sigmoid-constrained block parameters onto the per-unit vector.

    ``heart`` and ``brain`` are already in alpha space (base-10 sigmoid of
    the block logits). A missing side keeps that slice of ``shared``, so a
    blank block still follows the shared MT-RNN list. Lists shorter than
    the block are tiled; a range is already one value per unit.
    """
    if heart is None and brain is None:
        return shared
    if heart is None:
        heart_piece = shared[: split.heart]
    else:
        heart_piece = _tile_tensor(heart, split.heart).to(
            dtype=shared.dtype, device=shared.device
        )
    if brain is None:
        brain_piece = shared[split.heart :]
    else:
        brain_piece = _tile_tensor(brain, split.brain).to(
            dtype=shared.dtype, device=shared.device
        )
    return torch.cat((heart_piece, brain_piece), dim=0)


def apply_block_alphas(shared: torch.Tensor, split: HiddenBlockSplit) -> torch.Tensor:
    """Replace a block with its list (tiled) or its min,max range.

    A blank block keeps that slice of ``shared``. These values are
    constants. The trainable path is ``apply_trainable_block_alphas``,
    used when ``optimize_alphas`` is on.
    """
    out = shared
    replacements = (
        (slice(0, split.heart), split.heart_alphas, split.heart_alpha_range, split.heart),
        (
            slice(split.heart, None),
            split.brain_alphas,
            split.brain_alpha_range,
            split.brain,
        ),
    )
    for span, values, bounds, width in replacements:
        if bounds is not None:
            piece = _range_values(bounds, width, shared)
        elif values is not None:
            piece = _tile_values(values, width, shared)
        else:
            continue
        if out is shared:
            out = shared.clone()
        out[span] = piece
    return out


def masked_vanilla_step(cell, kind: str, feature_xt, h_t, c_t, block_mask):
    """One RNN or LSTM step with ``weight_hh`` multiplied by the block mask.

    The mask reads the hidden state that enters the step (``h_t``). It does
    not zero the updated state afterwards, and it does not touch ``weight_ih``
    or the bias, so an empty block can still move through the input. The
    parameters are multiplied in the forward rather than overwritten each
    step, so an optimizer cannot refill a hard-zeroed entry. Those entries
    are also written to 0 once at init.
    """
    x = feature_xt[0]
    h = h_t[0]
    if kind == "RNN":
        mixed = F.linear(x, cell.weight_ih_l0, cell.bias_ih_l0) + F.linear(
            h, cell.weight_hh_l0 * block_mask, cell.bias_hh_l0
        )
        activation = cell.nonlinearity if hasattr(cell, "nonlinearity") else "tanh"
        if activation == "relu":
            h_new = torch.relu(mixed)
        else:
            h_new = torch.tanh(mixed)
        return h_new.unsqueeze(0), None
    if kind == "LSTM":
        mask4 = block_mask.repeat(4, 1)
        gates = F.linear(x, cell.weight_ih_l0, cell.bias_ih_l0) + F.linear(
            h, cell.weight_hh_l0 * mask4, cell.bias_hh_l0
        )
        inp, forget, cell_gate, out_gate = gates.chunk(4, dim=-1)
        c_prev = c_t[0]
        c_new = torch.sigmoid(forget) * c_prev + torch.sigmoid(inp) * torch.tanh(
            cell_gate
        )
        h_new = torch.sigmoid(out_gate) * torch.tanh(c_new)
        return h_new.unsqueeze(0), c_new.unsqueeze(0)
    raise ValueError(f"masked step supports RNN and LSTM, got {kind!r}.")


def _zero_where_mask_is_hard_zero(parameter, mask) -> None:
    """Store 0 where the forward mask is exactly 0. Leave a weak coupling."""
    parameter.data.masked_fill_(mask == 0, 0)


def _zero_hard_masked_weights(module) -> None:
    """Init-time copy of the hard mask, so a checkpoint shows the structure.

    The forward still multiplies by the mask. A hard zero therefore stays
    zero under Adam, including AdamW, because the gradient on that entry
    is zero and weight decay of zero is zero. A weak-coupling factor is
    not a hard zero: those entries stay at their init and the forward
    scales them.
    """
    kind = module.type_rnn
    if kind == "PLRNN":
        # W_mask already zeros the diagonal. The block mask zeros the
        # off-block when coupling is 0. Multiply, do not replace W_mask.
        combined = module.rnn.W_mask * module.rnn.block_hh_mask
        _zero_where_mask_is_hard_zero(module.rnn.W, combined)
    elif kind == "shPLRNN":
        _zero_where_mask_is_hard_zero(module.rnn.W1, module.rnn.block_w1_mask)
        _zero_where_mask_is_hard_zero(module.rnn.W2, module.rnn.block_w2_mask)
    elif kind == "LSTM":
        _zero_where_mask_is_hard_zero(
            module.rnn.weight_hh_l0, module.block_hh_mask.repeat(4, 1)
        )
    elif kind == "RNN":
        _zero_where_mask_is_hard_zero(module.rnn.weight_hh_l0, module.block_hh_mask)
    _zero_where_mask_is_hard_zero(module.gen_out.weight, module.decoder_mask)


def install_hidden_block(module) -> None:
    """Attach masks after the cell and the decoder exist. Off stores nothing."""
    split = parse_hidden_block_split(module.cfg, module.dim_rnn, module.y_dim)
    module.hidden_block = split
    if split is None:
        return
    if module.num_rnn != 1:
        raise ValueError(
            "hidden_block_split supports num_rnn = 1, got "
            f"{module.num_rnn}."
        )
    if module.dense_h_x:
        raise ValueError(
            "hidden_block_split reads the hidden state with gen_out. "
            "dense_h_x must be empty."
        )
    if module.type_rnn not in {"RNN", "LSTM", "PLRNN", "shPLRNN"}:
        raise ValueError(
            "hidden_block_split supports RNN, LSTM, PLRNN, and shPLRNN, "
            f"got {module.type_rnn!r}."
        )
    module.register_buffer(
        "decoder_mask",
        decoder_mask(split.heart, split.brain, module.y_dim, split.ecg, split.eeg),
    )
    if module.type_rnn in {"RNN", "LSTM", "PLRNN"}:
        mask = recurrent_block_mask(split.heart, split.brain, split.coupling)
        if module.type_rnn == "PLRNN":
            module.rnn.register_buffer("block_hh_mask", mask)
        else:
            module.register_buffer("block_hh_mask", mask)
    else:
        heart_sh, brain_sh = shallow_split_sizes(
            split.heart, split.brain, module.hidden_sh_size
        )
        w1, w2 = shallow_factor_masks(
            split.heart, split.brain, heart_sh, brain_sh, split.coupling
        )
        module.rnn.register_buffer("block_w1_mask", w1)
        module.rnn.register_buffer("block_w2_mask", w2)
        module.rnn.heart_sh = heart_sh
        module.rnn.brain_sh = brain_sh
    _zero_hard_masked_weights(module)
