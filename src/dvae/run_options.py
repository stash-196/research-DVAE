"""Training and eval switches that stay off unless a config or CLI asks.

Blank, ``None``, and ``off`` keep today's reconstruction validation, checkpoint
cadence, and in-domain eval paths.
"""

from __future__ import annotations

import math
import re

_OFF = {"", "none", "null", "off", "false", "no"}
_VAL_CRITERIA = ("recon", "spectrum")


def _text(raw) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def parse_val_criteria(raw) -> tuple[str, ...]:
    """Comma-separated validation criteria. Default is ``('recon',)``.

    ``recon`` is the reconstruction validation loss. ``spectrum`` is the
    free-run spectral error. Blank, ``None``, and ``off`` mean ``recon``.

    One entry is used raw, so ``recon`` keeps the historical ``val_loss``
    comparison. Two entries are added after each is divided by its own
    value at the first validation epoch. That sum is lower-is-better.
    Each first-epoch value must be finite and non-zero.
    """
    text = _text(raw)
    if text.lower() in _OFF:
        return ("recon",)
    parts = [part.strip().lower() for part in text.split(",")]
    parts = [part for part in parts if part]
    if not parts:
        return ("recon",)
    unknown = [part for part in parts if part not in _VAL_CRITERIA]
    if unknown:
        raise ValueError(
            "val_criteria entries must be 'recon' and/or 'spectrum', "
            f"got {raw!r}."
        )
    if len(parts) != len(set(parts)):
        raise ValueError(f"val_criteria repeats an entry, got {raw!r}.")
    return tuple(parts)


def normalised_criterion_sum(components: dict, reference: dict) -> float:
    """Sum of ``component / reference`` in ``reference`` key order.

    Used only when more than one criterion is selected. ``reference`` is
    each criterion's value at the first validation epoch.
    """
    if not reference:
        raise ValueError("val criteria reference is empty.")
    total = 0.0
    for name, ref in reference.items():
        if name not in components:
            raise ValueError(f"val criterion {name!r} is missing from this epoch.")
        ref_value = float(ref)
        current = float(components[name])
        if (not math.isfinite(ref_value)) or ref_value == 0.0:
            raise ValueError(
                f"val criterion {name!r} at the first validation epoch must be "
                f"finite and non-zero so it can normalise later epochs, "
                f"got {ref!r}."
            )
        if not math.isfinite(current):
            raise ValueError(
                f"val criterion {name!r} is not finite, got {components[name]!r}."
            )
        total += current / ref_value
    return total


def criteria_log_fields(criteria, val_spectrum, val_criteria_score, reference, end):
    """Extra checkpoint keys. Empty when the only criterion is ``recon``."""
    names = tuple(criteria)
    if names == ("recon",):
        return {}
    stop = int(end)
    fields = {"val_criteria": ",".join(names)}
    if "spectrum" in names and val_spectrum is not None:
        fields["val_spectrum"] = val_spectrum[:stop]
    if len(names) > 1:
        fields["val_criteria_score"] = val_criteria_score[:stop]
        if reference:
            fields["val_criteria_reference"] = {
                name: float(reference[name]) for name in names
            }
    return fields


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
