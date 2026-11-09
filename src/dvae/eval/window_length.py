"""Cap requested windows to the longest subject, stay, or waveform segment.

Pooled ``len(seq)`` counts every subject. A window cannot cross that
boundary, so a request longer than one subject used to produce an empty
index and then crash the caller.
"""

from __future__ import annotations

from dvae.dataset.xhro_proper_dataset import (
    dataset_max_window_length,
    max_group_span,
)

# Shortest metrics window auto-fit will shrink to. Below this, spectrum
# bins are not worth scoring; a requested window that already fits is
# kept even when it is shorter.
METRICS_WINDOW_FLOOR = 4

__all__ = [
    "METRICS_WINDOW_FLOOR",
    "apply_eval_seq_len",
    "block_len_after_autofit",
    "cap_eval_seq_len",
    "choose_metrics_window_length",
    "dataset_max_window_length",
    "max_group_span",
    "subject_window_counts",
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


def choose_metrics_window_length(
    requested,
    counts_at,
    *,
    floor: int = METRICS_WINDOW_FLOOR,
    allow_shrink: bool = True,
) -> dict:
    """Pick a metrics window length from per-subject window counts.

    ``counts_at(length)`` returns one non-negative count per subject (or
    a single count when the series is one piece). The requested length
    is kept when it already yields at least one window.

    When it yields none and ``allow_shrink`` is true, the result is the
    longest length ``>= floor`` that maximizes how many subjects have at
    least one window. Lengths below ``floor`` are not considered. If
    every candidate is empty, ``length`` is ``None`` and ``reason`` is
    ``no_windows``. Explicit CLI lengths pass ``allow_shrink=False`` and
    then keep the old skip instead of shrinking.
    """
    requested = int(requested)
    floor = int(floor)
    empty = {
        "length": None,
        "shrunk": False,
        "reason": "no_windows",
        "counts": [],
        "from_length": requested,
        "subjects_covered": 0,
        "n_subjects": 0,
    }
    if requested < 1:
        return empty
    requested_counts = [int(count) for count in list(counts_at(requested))]
    covered = sum(1 for count in requested_counts if count >= 1)
    if sum(requested_counts) > 0:
        return {
            "length": requested,
            "shrunk": False,
            "reason": None,
            "counts": requested_counts,
            "from_length": requested,
            "subjects_covered": covered,
            "n_subjects": len(requested_counts),
        }
    if not allow_shrink or requested <= floor:
        return {
            "length": None,
            "shrunk": False,
            "reason": "no_windows",
            "counts": requested_counts,
            "from_length": requested,
            "subjects_covered": 0,
            "n_subjects": len(requested_counts),
        }
    best_len = None
    best_score = 0
    best_counts = None
    n_subjects = len(requested_counts)
    for length in range(requested - 1, floor - 1, -1):
        counts = [int(count) for count in list(counts_at(length))]
        score = sum(1 for count in counts if count >= 1)
        # Scanning long to short: the first time a score is seen it is
        # the longest length that achieves it. Ties keep that length.
        if score > best_score:
            best_score = score
            best_len = int(length)
            best_counts = counts
            # Every subject this probe knows about already has a window.
            # Shorter lengths cannot raise the score, so stop.
            if n_subjects and score >= n_subjects and len(counts) <= n_subjects:
                break
    if best_len is None or best_score < 1:
        return {
            "length": None,
            "shrunk": False,
            "reason": "no_windows",
            "counts": requested_counts,
            "from_length": requested,
            "subjects_covered": 0,
            "n_subjects": len(requested_counts),
        }
    return {
        "length": best_len,
        "shrunk": True,
        "reason": None,
        "counts": best_counts,
        "from_length": requested,
        "subjects_covered": best_score,
        "n_subjects": len(best_counts),
    }


def subject_window_counts(dataset) -> dict:
    """Window counts keyed by subject, in first-seen order.

    A series with no subject, stay, or segment index uses the key
    ``series``. Subjects that currently have no window are included
    with count 0. Starts outside the group index are ``out_of_range``.
    """
    group = dataset_row_group(dataset)
    starts = [int(start) for start in (getattr(dataset, "data_idx", None) or [])]
    if group is None:
        return {"series": len(starts)}
    order = []
    seen = set()
    for value in group:
        key = str(value)
        if key not in seen:
            seen.add(key)
            order.append(key)
    counts = {key: 0 for key in order}
    n_rows = len(order) if not hasattr(group, "__len__") else len(group)
    for start in starts:
        if 0 <= start < n_rows:
            key = str(group[start])
            counts[key] = counts.get(key, 0) + 1
        else:
            counts["out_of_range"] = counts.get("out_of_range", 0) + 1
    return counts


def block_len_after_autofit(seq_len, requested_block, *, shrunk: bool) -> int:
    """Half the window after a shrink, so half_half splits that window.

    An unshrunk run keeps the requested block length, including an
    explicit ``--auto-eval-block-len``.
    """
    if not shrunk:
        return int(requested_block)
    return max(1, int(seq_len) // 2)
