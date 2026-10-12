"""Training and eval switches.

``val_criteria`` defaults to recon and spectrum. A blank key keeps that
pair: each term is divided by its first validation epoch and the sum is
the selection score. ``[DataFrame] legacy_defaults`` true, or the same
switch under its old name ``legacy_windowing``, keeps reconstruction-only
selection when ``val_criteria`` is blank. An explicit ``val_criteria =
recon`` does that on its own.
Checkpoint cadence and the in-domain eval path stay off unless set.
"""

from __future__ import annotations

import math
import re

_OFF = {"", "none", "null", "off", "false", "no"}
_VAL_UNSET = {"", "none", "null"}
_VAL_CRITERIA = ("recon", "spectrum")
DEFAULT_VAL_CRITERIA = ("recon", "spectrum")


def _text(raw) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def legacy_windowing_enabled(raw) -> bool:
    """True only for an explicit true/yes/on. Blank stays false."""
    return _text(raw).lower() in {"1", "true", "yes", "y", "on"}


def legacy_defaults_from_config(cfg) -> bool:
    """True when ``legacy_defaults`` or ``legacy_windowing`` is explicitly on.

    ``legacy_defaults`` is the name to write. ``legacy_windowing`` is the
    same switch. Blank on both is false. An explicit true on either name
    restores the previous defaults for the keys that are still blank.
    """
    if not cfg.has_section("DataFrame"):
        return False
    for option in ("legacy_defaults", "legacy_windowing"):
        if not cfg.has_option("DataFrame", option):
            continue
        if legacy_windowing_enabled(cfg.get("DataFrame", option)):
            return True
    return False


def parse_val_criteria(raw, legacy_windowing=False) -> tuple[str, ...]:
    """Comma-separated validation criteria.

    Blank and ``None`` are ``recon,spectrum``. ``legacy_windowing`` true
    (the ``legacy_defaults`` switch) changes that blank default back to
    ``recon``. An explicit ``recon`` is always recon only, and an
    explicit list is kept as written.

    ``recon`` is the reconstruction validation loss. ``spectrum`` is the
    free-run spectral error. One entry is used raw, so ``recon`` keeps
    the historical ``val_loss`` comparison. Two entries are added after
    each is divided by its own value at the first validation epoch. That
    sum is lower-is-better. Each first-epoch value must be finite and
    non-zero.
    """
    text = _text(raw)
    lowered = text.lower()
    if lowered in _VAL_UNSET:
        if legacy_windowing:
            return ("recon",)
        return DEFAULT_VAL_CRITERIA
    if lowered in {"off", "false", "no", "recon"}:
        return ("recon",)
    parts = [part.strip().lower() for part in text.split(",")]
    parts = [part for part in parts if part]
    if not parts:
        if legacy_windowing:
            return ("recon",)
        return DEFAULT_VAL_CRITERIA
    unknown = [part for part in parts if part not in _VAL_CRITERIA]
    if unknown:
        raise ValueError(
            "val_criteria entries must be 'recon' and/or 'spectrum', "
            f"got {raw!r}."
        )
    if len(parts) != len(set(parts)):
        raise ValueError(f"val_criteria repeats an entry, got {raw!r}.")
    return tuple(parts)


def resolve_val_criteria(cfg) -> tuple[str, ...]:
    """Read ``[Training] val_criteria`` and the legacy-defaults switch.

    A missing ``val_criteria`` key falls back to the older ``val_criterion``
    name. Blank uses the default pair unless ``legacy_defaults`` or
    ``legacy_windowing`` is on.
    """
    if cfg.has_option("Training", "val_criteria"):
        raw = cfg.get("Training", "val_criteria")
    elif cfg.has_option("Training", "val_criterion"):
        raw = cfg.get("Training", "val_criterion")
    else:
        raw = None
    return parse_val_criteria(raw, legacy_windowing=legacy_defaults_from_config(cfg))


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
