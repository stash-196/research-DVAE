"""Structured training-log lines and the bump-event table.

The lines are logged every epoch (timing and selection) or on each
warm-up bump and the final stop. The bump table is what a later redraw
uses to place the vertical lines: one row per changed knob.
"""

from __future__ import annotations

# Per-epoch seconds. Present in every new history file.
TIMING_SERIES_KEYS = (
    "time_train",
    "time_val",
    "time_val_forward",
    "time_spectrum",
    "time_drift",
    "time_rosenstein",
    "time_plot",
    "time_eval",
)

# Stack order on the training figure. Forward is the shared model cost
# of the fixed-window metrics; spectrum, drift, and Rosenstein are the
# three pieces after that.
TIMING_PLOT_LABELS = (
    ("time_train", "train"),
    ("time_val", "val"),
    ("time_val_forward", "metrics forward"),
    ("time_spectrum", "spectrum"),
    ("time_drift", "drift"),
    ("time_rosenstein", "rosenstein"),
    ("time_plot", "plot"),
    ("time_eval", "eval"),
)

_BUMP_TYPES = ("window", "auto", "noise", "KL")


def format_timing_line(epoch, times) -> str:
    """One wall-time line. Keys missing from ``times`` are logged as 0."""
    parts = ["Epoch: {} wall_s".format(int(epoch))]
    for key, label in TIMING_PLOT_LABELS:
        parts.append("{}={:.4f}".format(label.replace(" ", "_"), float(times.get(key, 0.0))))
    return " ".join(parts)


def _component_text(raw, norm) -> str:
    pieces = []
    for name, value in raw.items():
        if norm and name in norm:
            pieces.append(
                "{} raw={:.6f} norm={:.6f}".format(name, float(value), float(norm[name]))
            )
        else:
            pieces.append("{} raw={:.6f} norm=off".format(name, float(value)))
    return ", ".join(pieces)


def format_selection_line(epoch, score, raw, norm, improved) -> str:
    """Score, each selected component raw and normalised, and improved."""
    flag = "yes" if improved else "no"
    return (
        "Selection epoch {}: score={:.6f} improved={} components={}".format(
            int(epoch), float(score), flag, _component_text(raw, norm)
        )
    )


def format_new_best_line(epoch, score, raw, norm) -> str:
    """Logged when this epoch becomes the checkpoint."""
    return "New best epoch {}: score={:.6f}, components={}".format(
        int(epoch), float(score), _component_text(raw, norm)
    )


def format_bump_line(row) -> str:
    """One knob change. ``type`` is window, auto, noise, or KL."""
    return (
        "Warm-up bump epoch={epoch} type={type} old={old:.6g} new={new:.6g} "
        "patience_before={patience:.6g} patience_after={patience_after:.6g} "
        "cpt_patience={cpt_patience}".format(
            epoch=int(row["epoch"]),
            type=row["type"],
            old=float(row["old"]),
            new=float(row["new"]),
            patience=float(row["patience"]),
            patience_after=float(row["patience_after"]),
            cpt_patience=int(row["cpt_patience"]),
        )
    )


def format_early_stop_line(epoch, cpt_patience, active_patience) -> str:
    """Final stop: patience exceeded and every warm-up phase is finished."""
    return (
        "Early stop epoch={}: reason=patience cpt_patience={} "
        "active_patience={:.6g} warmups_finished=true".format(
            int(epoch), int(cpt_patience), float(active_patience)
        )
    )


def format_epoch_budget_line(epoch) -> str:
    """The loop finished the configured epoch count without an early stop."""
    return "Training finished epoch={}: reason=epoch_budget".format(int(epoch))


def make_bump_row(epoch, kind, old, new, patience_before, patience_after, cpt_patience):
    """One table row. ``patience`` is the active patience that fired the bump."""
    if kind not in _BUMP_TYPES:
        raise ValueError(
            "bump type must be window, auto, noise, or KL, got {!r}.".format(kind)
        )
    return {
        "epoch": int(epoch),
        "type": kind,
        "old": float(old),
        "new": float(new),
        "patience": float(patience_before),
        "patience_after": float(patience_after),
        "cpt_patience": int(cpt_patience),
    }


def vertical_line_epochs(events, kind) -> list:
    """Epochs of the vertical lines for ``kind``, including the origin.

    The bump row's ``epoch`` is the epoch that decided the bump. The
    warm-up series stores the new value on the following epoch, and the
    figure draws the line there. Prepending 0 rebuilds the marker set
    ``warmup_change_epochs`` produces from that series.
    """
    epochs = [0]
    for row in events or []:
        if row.get("type") != kind:
            continue
        epoch = int(row["epoch"]) + 1
        if epoch not in epochs:
            epochs.append(epoch)
    return epochs
