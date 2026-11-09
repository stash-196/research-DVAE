"""Cap requested windows to the longest subject, stay, or waveform segment.

Pooled ``len(seq)`` counts every subject. A window cannot cross that
boundary, so a request longer than one subject used to produce an empty
index and then crash the caller.
"""

from __future__ import annotations

from dvae.dataset.xhro_proper_dataset import (
    dataset_max_window_length,
    dataset_row_group,
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


def _probe_counts(length, result):
    """``(stored seq_len, count list)`` from a probe result.

    A bare list or dict means the cutter kept ``length``. A pair is
    ``(stored seq_len, counts)``. Counts may be a dict of per-subject
    totals; only the values are scored.
    """
    stored = int(length)
    counts = result
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and not isinstance(result[0], (list, tuple, dict))
    ):
        stored = int(result[0])
        counts = result[1]
    if isinstance(counts, dict):
        count_list = [int(value) for value in counts.values()]
    else:
        count_list = [int(value) for value in list(counts)]
    return stored, count_list


def choose_metrics_window_length(
    requested,
    counts_at,
    *,
    floor: int = METRICS_WINDOW_FLOOR,
    allow_shrink: bool = True,
    max_length=None,
) -> dict:
    """Pick a metrics window length from per-subject window counts.

    ``counts_at(length)`` returns one non-negative count per subject, or
    ``(stored_seq_len, counts)`` when the cutter may shorten the request.
    A request that yields a window at the length it asked for is kept.
    A request that yields windows only because the cutter stored a
    shorter ``seq_len`` is a shrink: ``length`` is that stored value and
    ``shrunk`` is true when shrinking is allowed. An explicit length
    (``allow_shrink=False``) still reports the stored length and does
    not set ``shrunk``, so the block length stays at the request.

    When the request yields none and ``allow_shrink`` is true, the
    result is the longest length ``>= floor`` that maximizes how many
    subjects have at least one window. The downward scan starts at
    ``max_length`` (the longest group span) when that is shorter than
    the request, and a probe that stored a shorter ``seq_len`` is not
    repeated for the lengths in between. Lengths below ``floor`` are
    not considered. If every candidate is empty, ``length`` is ``None``
    and ``reason`` is ``no_windows``.
    """
    requested = int(requested)
    floor = int(floor)
    cache = {}
    cap_ranges = []

    def probe(length):
        length = int(length)
        cached = cache.get(length)
        if cached is None:
            for low, high in cap_ranges:
                if low <= length <= high:
                    cached = cache[low]
                    break
        if cached is not None:
            return cached
        stored, counts = _probe_counts(length, counts_at(length))
        cached = (stored, counts)
        cache[length] = cached
        cache[stored] = cached
        if stored < length:
            cap_ranges.append((stored, length))
        return cached

    def decision(length, shrunk, reason, counts, stored):
        covered = sum(1 for count in counts if count >= 1)
        return {
            "length": length,
            "shrunk": bool(shrunk),
            "reason": reason,
            "counts": list(counts),
            "from_length": requested,
            "subjects_covered": 0 if length is None else covered,
            "n_subjects": len(counts),
            "stored_length": int(stored),
        }

    if requested < 1:
        return decision(None, False, "no_windows", [], 0)
    stored, requested_counts = probe(requested)
    if sum(requested_counts) > 0:
        use = stored if stored > 0 else requested
        shrunk = bool(allow_shrink and stored < requested)
        return decision(int(use), shrunk, None, requested_counts, stored)
    if not allow_shrink or requested <= floor:
        return decision(None, False, "no_windows", requested_counts, stored)

    # The request itself produced nothing. Do not walk every integer
    # above the group span, and do not repeat a capped recut.
    scan_hi = requested - 1
    if max_length is not None:
        scan_hi = min(scan_hi, int(max_length))
    if stored < requested:
        scan_hi = min(scan_hi, stored - 1)
    best_len = None
    best_score = 0
    best_counts = None
    best_stored = stored
    n_subjects = len(requested_counts)
    length = int(scan_hi)
    while length >= floor:
        got_stored, counts = probe(length)
        if len(counts) > n_subjects:
            n_subjects = len(counts)
        score = sum(1 for count in counts if count >= 1)
        # Scanning long to short: the first time a score is seen it is
        # the longest length that achieves it. Ties keep that length.
        # A cutter cap is scored at the stored length, not the request.
        candidate = got_stored if got_stored > 0 else length
        if score > best_score:
            best_score = score
            best_len = int(candidate)
            best_counts = counts
            best_stored = got_stored
            if n_subjects and score >= n_subjects and len(counts) <= n_subjects:
                break
        if got_stored < length:
            if got_stored < floor:
                break
            length = got_stored - 1
        else:
            length -= 1
    if best_len is None or best_score < 1:
        return decision(None, False, "no_windows", requested_counts, stored)
    return decision(best_len, True, None, best_counts, best_stored)


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
