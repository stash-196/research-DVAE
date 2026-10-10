"""Training and eval switches that stay off unless a config or CLI asks.

Blank, ``None``, and ``off`` keep today's reconstruction validation, checkpoint
cadence, and in-domain eval paths.
"""

from __future__ import annotations

import re

_OFF = {"", "none", "null", "off", "false", "no"}


def _text(raw) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def parse_val_criterion(raw) -> str:
    """``recon`` (default) or ``spectrum``.

    Spectrum is the only other accepted value. It scores the free-run half
    alone and is not blended with the reconstruction loss.
    """
    text = _text(raw).lower()
    if text in _OFF or text == "recon":
        return "recon"
    if text == "spectrum":
        return "spectrum"
    raise ValueError(
        f"val_criterion must be 'recon' or 'spectrum', got {raw!r}."
    )


def parse_optional_int(raw, default: int, name: str) -> int:
    """Blank and ``None`` keep ``default``. Any other token must be an int."""
    text = _text(raw).lower()
    if text in _OFF:
        return int(default)
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc


def parse_checkpoint_every_n_epochs(raw) -> int:
    """Extra current-epoch checkpoints. ``0`` / blank / ``off`` means none."""
    text = _text(raw).lower()
    if text in _OFF or text == "0":
        return 0
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"checkpoint_every_n_epochs must be a non-negative integer "
            f"or off, got {raw!r}."
        ) from exc
    if value < 0:
        raise ValueError(
            f"checkpoint_every_n_epochs must be >= 0, got {value}."
        )
    return value


def _norm_label(raw) -> str:
    text = _text(raw)
    if text.lower() in ("", "none", "null"):
        return ""
    return text


def _sanitize_suffix(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "override"


def resolve_mask_label_override(
    config_label,
    override,
    explicit_suffix=None,
) -> tuple[str | None, str | None]:
    """Return ``(mask_label for the dataset, output suffix or None)``.

    No CLI override keeps the config label and writes the usual files.
    An override that matches the config also keeps the usual files.
    A different variant gets a suffix so ``evaluation_summary.yaml`` and
    ``post_training_figs`` at the run root are left in place.
    """
    if _text(override) == "":
        return config_label, None
    if _norm_label(override) == _norm_label(config_label):
        return override, None
    suffix_raw = explicit_suffix if _text(explicit_suffix) else override
    return override, _sanitize_suffix(str(suffix_raw))


def eval_output_dir(weights_dir: str, suffix: str | None) -> str:
    """Run directory for eval artifacts. ``suffix`` nests them under ``eval_mask_``."""
    if suffix is None or _text(suffix) == "":
        return weights_dir
    return f"{weights_dir.rstrip('/')}/eval_mask_{_sanitize_suffix(suffix)}"
