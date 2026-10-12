"""Optional subband spectrum loss and decorrelation penalty.

Both are off when the weight is blank, 0, or off. A warm-up flag ramps
the weight with the autonomous and KL bumps:

    done / (autonomous_steps + kl_steps)

``done`` is how many of those bumps have already finished before this
epoch's loss. The bump that fires at the end of an epoch increments the
count afterwards, so the trigger epoch still uses the previous scale.
Window-length bumps do not advance ``done``. The ratio is 1 when the
warm-up flag is off or the denominator is 0.

Channel lists and ``subband_low_hz`` do not treat ``0`` as off. Blank,
``off``, and ``none`` are the unset tokens. ``0`` is channel 0 or 0 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_OFF = {"", "none", "null", "off", "false", "no", "n", "0", "0.0"}
# Channel lists and band edges. "0" is a real index or 0 Hz, not off.
_UNSET = {"", "off", "none"}
_ON = {"1", "true", "yes", "y", "on"}


def _text(raw) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def _option(cfg, option: str):
    if not cfg.has_section("Training"):
        return None
    if not cfg.has_option("Training", option):
        return None
    return cfg.get("Training", option)


def parse_loss_weight(raw, name: str) -> float:
    """Blank, 0, and off are 0. Anything else is a positive finite weight."""
    text = _text(raw).lower()
    if text in _OFF:
        return 0.0
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a weight, 0, or off, got {raw!r}.") from exc
    if not (value > 0.0) or value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be positive when set, got {raw!r}.")
    return value


def _flag(raw, name: str) -> bool:
    text = _text(raw).lower()
    if text in _OFF or text == "":
        return False
    if text in _ON:
        return True
    raise ValueError(f"{name} must be true or false, got {raw!r}.")


def _channels(raw, name: str, n_channels: int) -> tuple[int, ...]:
    text = _text(raw)
    if text.lower() in _UNSET:
        raise ValueError(f"{name} is required when the loss weight is on.")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{name} is required when the loss weight is on.")
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


def _optional_float(raw, name: str) -> float | None:
    text = _text(raw).lower()
    if text in _UNSET:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}.") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite, got {raw!r}.")
    return value


@dataclass(frozen=True)
class RegularizerSettings:
    subband_weight: float = 0.0
    subband_warmup: bool = False
    subband_channels: tuple[int, ...] = ()
    subband_fs: float = 0.0
    subband_low_hz: float = 0.0
    subband_high_hz: float | None = None
    decorr_weight: float = 0.0
    decorr_warmup: bool = False
    decorr_mode: str = "hidden"
    decorr_groups: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    @property
    def subband_on(self) -> bool:
        return self.subband_weight > 0.0

    @property
    def decorr_on(self) -> bool:
        return self.decorr_weight > 0.0


def parse_regularizer_settings(cfg, y_dim: int, hidden_block) -> RegularizerSettings:
    sub_w = parse_loss_weight(_option(cfg, "subband_loss_weight"), "subband_loss_weight")
    dec_w = parse_loss_weight(_option(cfg, "decorr_loss_weight"), "decorr_loss_weight")
    sub_channels: tuple[int, ...] = ()
    fs = 0.0
    low = 0.0
    high = None
    if sub_w > 0.0:
        sub_channels = _channels(
            _option(cfg, "subband_channels"), "subband_channels", int(y_dim)
        )
        fs_raw = _optional_float(_option(cfg, "subband_fs"), "subband_fs")
        low_raw = _optional_float(_option(cfg, "subband_low_hz"), "subband_low_hz")
        if fs_raw is None or fs_raw <= 0.0:
            raise ValueError("subband_fs must be a positive sample rate in Hz.")
        if low_raw is None or low_raw < 0.0:
            raise ValueError("subband_low_hz must be >= 0 when the subband loss is on.")
        high = _optional_float(_option(cfg, "subband_high_hz"), "subband_high_hz")
        if high is not None and high <= low_raw:
            raise ValueError(
                f"subband_high_hz must be above subband_low_hz, got {high} <= {low_raw}."
            )
        fs = fs_raw
        low = low_raw
    mode = _text(_option(cfg, "decorr_mode")).lower() or "hidden"
    groups = None
    if dec_w > 0.0:
        if mode not in {"hidden", "output"}:
            raise ValueError(
                f"decorr_mode must be hidden or output, got {mode!r}."
            )
        if mode == "hidden":
            if hidden_block is None:
                raise ValueError(
                    "decorr_mode hidden needs hidden_block_split on, so the "
                    "heart and brain blocks are defined."
                )
        else:
            groups = _output_groups(_option(cfg, "decorr_groups"), int(y_dim))
    return RegularizerSettings(
        subband_weight=sub_w,
        subband_warmup=_flag(_option(cfg, "subband_loss_warmup"), "subband_loss_warmup")
        if sub_w > 0.0
        else False,
        subband_channels=sub_channels,
        subband_fs=fs,
        subband_low_hz=low,
        subband_high_hz=high,
        decorr_weight=dec_w,
        decorr_warmup=_flag(_option(cfg, "decorr_loss_warmup"), "decorr_loss_warmup")
        if dec_w > 0.0
        else False,
        decorr_mode=mode if dec_w > 0.0 else "hidden",
        decorr_groups=groups,
    )


def _output_groups(raw, n_channels: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    text = _text(raw)
    sides = [part.strip() for part in text.split("|")]
    if len(sides) != 2:
        raise ValueError(
            "decorr_groups must be two 0-based groups separated by |, "
            f"for example 0,1|2,3, got {raw!r}."
        )
    left = _channels(sides[0], "decorr_groups", n_channels)
    right = _channels(sides[1], "decorr_groups", n_channels)
    overlap = set(left) & set(right)
    if overlap:
        raise ValueError(
            f"decorr_groups overlap on {sorted(overlap)}, got {raw!r}."
        )
    return left, right


def warmup_progress(
    auto_kl_done: int, autonomous_steps: int, kl_steps: int, warmup_on: bool
) -> float:
    """0 at the start of a warm-up, 1 when it is off or there is nothing to ramp."""
    if not warmup_on:
        return 1.0
    total = int(autonomous_steps) + int(kl_steps)
    if total <= 0:
        return 1.0
    done = max(int(auto_kl_done), 0)
    return min(done / total, 1.0)


def effective_weight(weight: float, warmup_on: bool, auto_kl_done, autonomous_steps, kl_steps) -> float:
    return float(weight) * warmup_progress(
        auto_kl_done, autonomous_steps, kl_steps, warmup_on
    )


def _band_mask(n_time: int, fs: float, low_hz: float, high_hz: float | None, device, dtype):
    freqs = torch.fft.rfftfreq(n_time, d=1.0 / float(fs)).to(device=device)
    high = float(fs) / 2.0 if high_hz is None else float(high_hz)
    mask = (freqs >= float(low_hz)) & (freqs <= high + 1e-8)
    if int(mask.sum()) < 1:
        raise ValueError(
            f"subband {low_hz}..{high} Hz contains no rFFT bins at fs={fs} "
            f"and length {n_time}."
        )
    return mask


def subband_power_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    channels: tuple[int, ...],
    fs: float,
    low_hz: float,
    high_hz: float | None,
) -> torch.Tensor:
    """MSE of band-normalised rFFT power on the chosen channels.

    ``predicted`` and ``target`` are ``(time, batch, channel)``. Power
    inside the band is divided by its own sum, per series and channel.
    """
    pred = torch.nan_to_num(predicted[:, :, list(channels)], nan=0.0)
    gt = torch.nan_to_num(target[:, :, list(channels)], nan=0.0)
    band = _band_mask(pred.shape[0], fs, low_hz, high_hz, pred.device, pred.dtype)
    pred_power = torch.fft.rfft(pred, dim=0).abs().pow(2)[band]
    gt_power = torch.fft.rfft(gt, dim=0).abs().pow(2)[band]
    pred_norm = pred_power / pred_power.sum(dim=0, keepdim=True).clamp_min(1e-8)
    gt_norm = gt_power / gt_power.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return (pred_norm - gt_norm).pow(2).mean()


def _standardize(series: torch.Tensor) -> torch.Tensor:
    """Zero mean and unit variance over time. ``series`` is (time, batch, channel)."""
    centered = series - series.mean(dim=0, keepdim=True)
    variance = centered.pow(2).mean(dim=0, keepdim=True)
    return centered / torch.sqrt(variance + 1e-8)


def _cross_corr_sq(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Mean squared correlation between every left channel and every right channel."""
    left_z = _standardize(torch.nan_to_num(left, nan=0.0))
    right_z = _standardize(torch.nan_to_num(right, nan=0.0))
    corr = torch.einsum("tbi,tbj->bij", left_z, right_z) / float(left_z.shape[0])
    return corr.pow(2).mean()


def decorr_hidden_loss(hidden: torch.Tensor, heart: int, brain: int) -> torch.Tensor:
    """Cross-correlation between the heart and brain hidden blocks."""
    return _cross_corr_sq(hidden[:, :, :heart], hidden[:, :, heart : heart + brain])


def decorr_output_loss(
    output: torch.Tensor,
    group_a: tuple[int, ...],
    group_b: tuple[int, ...],
) -> torch.Tensor:
    """Off-diagonal block between two output channel groups."""
    return _cross_corr_sq(output[:, :, list(group_a)], output[:, :, list(group_b)])


def regularizer_losses(model, target: torch.Tensor, settings: RegularizerSettings):
    """Raw subband and decorr terms. A disabled term is a zero on ``target``."""
    zero = target.new_zeros(())
    sub = zero
    dec = zero
    if settings.subband_on:
        sub = subband_power_loss(
            model.y,
            target,
            settings.subband_channels,
            settings.subband_fs,
            settings.subband_low_hz,
            settings.subband_high_hz,
        )
    if settings.decorr_on:
        if settings.decorr_mode == "hidden":
            block = model.hidden_block
            dec = decorr_hidden_loss(model.h, block.heart, block.brain)
        else:
            groups = settings.decorr_groups
            dec = decorr_output_loss(model.y, groups[0], groups[1])
    return sub, dec


OPTIONAL_HISTORY_KEYS = (
    "train_subband",
    "val_subband",
    "subband_loss_weight_values",
    "train_decorr",
    "val_decorr",
    "decorr_loss_weight_values",
)
