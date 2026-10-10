"""Cap requested windows to the longest subject or stay.

Pooled ``len(seq)`` counts every subject. A window cannot cross that
boundary, so a request longer than one subject used to produce an empty
index and then crash the caller.
"""

from __future__ import annotations

from dvae.dataset.xhro_proper_dataset import (
    dataset_max_window_length,
    max_group_span,
)

__all__ = [
    "apply_eval_seq_len",
    "cap_eval_seq_len",
    "dataset_max_window_length",
    "max_group_span",
]


def cap_eval_seq_len(dataset, requested: int) -> int:
    """Min of the request and the longest in-subject span. ``0`` if there is no row."""
    span = dataset_max_window_length(dataset)
    if span < 1:
        return 0
    return int(min(int(requested), span))


def apply_eval_seq_len(dataset, requested: int) -> int:
    """Set ``seq_len`` to a length the dataset can actually window.

    Returns the effective length, or ``0`` when the window set is empty.
    Callers should skip visualization and metrics in that case instead of
    reading an unset batch.
    """
    length = cap_eval_seq_len(dataset, requested)
    if length < 1:
        return 0
    dataset.update_sequence_length(length)
    data_idx = getattr(dataset, "data_idx", None)
    if data_idx is None or len(data_idx) == 0:
        return 0
    return int(getattr(dataset, "seq_len", length))
