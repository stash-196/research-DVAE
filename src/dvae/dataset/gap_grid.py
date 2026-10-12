"""Native-grid gap keys written next to ``t`` / ``datetime_ns``.

Stages 04–09b may carry:

* ``present`` — bool per row. ``False`` is an inserted NaN row (lost packet).
* ``segment_id`` — per-row id. A change is a hard segment boundary.
* ``segment_bounds`` — ``(S, 2)`` half-open ``[start, end)`` pairs, or a 1-d
  edge array of length ``S + 1``. Used only when ``segment_id`` is absent.
* ``gap_indices`` — row indices of hard boundaries only (first row of the
  next piece). Redundant when ``segment_id`` already changes there.
* a per-gap table, one entry per gap: ``gap_t_start``, ``gap_n_missing``,
  ``gap_duration_s``, ``gap_filled``, ``gap_kind``, ``gap_index``.
  ``gap_t_start`` uses the same clock as ``t``. ``gap_index`` is the first
  inserted row of a filled gap, or the first row after a hard gap.

Old files omit these keys. Callers then keep today's window index rule.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .utils.data_utils import _boundary_cuts, longest_unbroken_span

PRESENT_COL = "__present"
SEGMENT_COL = "__segment_id"
GROUP_COL = "__window_group"
FILLED_RUN_COL = "__filled_run_sec"

_ROW_KEYS = ("present", "segment_id")
_TABLE_KEYS = (
    "gap_t_start",
    "gap_n_missing",
    "gap_duration_s",
    "gap_filled",
    "gap_kind",
    "gap_index",
)
_ALL_KEYS = _ROW_KEYS + ("segment_bounds", "gap_indices") + _TABLE_KEYS
_META_COLS = (PRESENT_COL, SEGMENT_COL, GROUP_COL)


def npz_has_gap_grid(npz) -> bool:
    files = npz.files
    return any(key in files for key in _ALL_KEYS)


def attach_gap_grid(df: pd.DataFrame, npz, max_fill_gap_sec) -> pd.DataFrame:
    """NaN out ``present=False`` rows and tag window groups. No-op if no keys."""
    if not npz_has_gap_grid(npz):
        return df
    n = len(df)
    t = np.asarray(npz["t"], dtype=np.float64) if "t" in npz.files else None
    present = _read_present(npz, n)
    segment_id = _read_segment_id(npz, n)
    if present is not None:
        _blank_absent_rows(df, present)
        df[PRESENT_COL] = present
    hard = _read_gap_indices(npz)
    spans = _long_filled_spans(
        npz, n, t, present, max_fill_gap_sec
    )
    if segment_id is not None:
        df[SEGMENT_COL] = segment_id
    if segment_id is not None or hard.size or spans:
        df[GROUP_COL] = window_group_ids(n, segment_id, hard, spans)
    if present is not None:
        df[FILLED_RUN_COL] = filled_run_seconds(n, present, t, npz)
    return df


def combine_gap_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Fold suffixed meta columns after an outer datetime merge.

    A row is present only when every stage that has the flag says so.
    A window group changes when either stage's group changes. Signal
    values are left as each stage blanked them: an ECG hole does not
    NaN a real EEG sample on the same timestamp.
    """
    out = df
    px, py = f"{PRESENT_COL}_x", f"{PRESENT_COL}_y"
    if px in out.columns or py in out.columns:
        left = _bool_side(out, px)
        right = _bool_side(out, py)
        out[PRESENT_COL] = left & right
        out = out.drop(columns=[c for c in (px, py) if c in out.columns])
    for name in (SEGMENT_COL, GROUP_COL):
        x, y = f"{name}_x", f"{name}_y"
        if x not in out.columns and y not in out.columns:
            continue
        ax = out[x] if x in out.columns else pd.Series(0, index=out.index)
        ay = out[y] if y in out.columns else pd.Series(0, index=out.index)
        codes = _factorize_pairs(ax, ay)
        out[name] = codes
        out = out.drop(columns=[c for c in (x, y) if c in out.columns])
    return _combine_filled_run(out)


def gap_load_stats(frame: pd.DataFrame) -> dict:
    """Rows, present fraction, segment runs, and window split points.

    No gap keys: every row counts as present, one segment, no split points.
    That is the old-file fallback. ``segments`` follows ``segment_id``
    (or bounds). ``split_points`` follows the window group, so a long
    filled gap adds cuts without pretending to be a new file segment.
    """
    n = int(len(frame))
    if PRESENT_COL in frame.columns and n:
        present_frac = float(np.mean(frame[PRESENT_COL].to_numpy(dtype=bool)))
    else:
        present_frac = 1.0
    if SEGMENT_COL in frame.columns and n:
        sid = frame[SEGMENT_COL].to_numpy()
        n_segments = 1 + int(np.sum(sid[1:] != sid[:-1])) if n > 1 else 1
    else:
        n_segments = 1 if n else 0
    if GROUP_COL in frame.columns and n > 1:
        gid = frame[GROUP_COL].to_numpy()
        n_splits = int(np.sum(gid[1:] != gid[:-1]))
    else:
        n_splits = 0
    return {
        "rows": n,
        "present_frac": present_frac,
        "segments": n_segments,
        "split_points": n_splits,
    }


def split_boundaries_from_ids(ids) -> np.ndarray:
    """Sample indices where the per-row id changes. Empty if there is one run."""
    ids = np.asarray(ids)
    if ids.size < 2:
        return np.asarray([], dtype=np.int64)
    return (np.flatnonzero(ids[1:] != ids[:-1]) + 1).astype(np.int64)


def filled_run_seconds(n, present, t, npz) -> np.ndarray:
    """Seconds of the filled-gap run each row belongs to. Present rows are 0.

    Contiguous ``present=False`` rows default to ``count * median(dt)``.
    A filled gap-table entry overwrites the rows it covers with
    ``gap_duration_s``.
    """
    n = int(n)
    dur = np.zeros(n, dtype=np.float64)
    if present is None or n == 0:
        return dur
    flags = np.asarray(present, dtype=bool).reshape(-1)
    dt = _median_positive_dt(t)
    i = 0
    while i < n:
        if flags[i]:
            i += 1
            continue
        j = i + 1
        while j < n and not flags[j]:
            j += 1
        dur[i:j] = (j - i) * dt
        i = j
    table = _gap_table(npz)
    if table is None or "gap_filled" not in table or "gap_duration_s" not in table:
        return dur
    filled = _as_bool(table["gap_filled"]).reshape(-1)
    duration = np.asarray(table["gap_duration_s"], dtype=np.float64).reshape(-1)
    n_missing = table.get("gap_n_missing")
    gap_index = table.get("gap_index")
    gap_t = table.get("gap_t_start")
    for i, (is_filled, gap_dur) in enumerate(zip(filled, duration)):
        if not bool(is_filled) or not np.isfinite(gap_dur):
            continue
        count = None
        if n_missing is not None and np.isfinite(np.asarray(n_missing).reshape(-1)[i]):
            count = int(np.asarray(n_missing).reshape(-1)[i])
        start = _locate_gap_start(
            n,
            t,
            None if gap_index is None else np.asarray(gap_index).reshape(-1)[i],
            None if gap_t is None else np.asarray(gap_t).reshape(-1)[i],
        )
        rows = _filled_rows(n, flags, start, count)
        if rows.size:
            dur[rows] = float(gap_dur)
    return dur


def recut_threshold_sec(max_fill_gap_sec, filled_nan_log_sec):
    """Seconds a filled NaN run must exceed to be counted at a re-cut.

    ``max_fill_gap_sec`` wins when it is set, because that is the split
    rule. Otherwise the log uses ``filled_nan_log_sec``. Both unset → no
    cutoff (the log prints ``na``).
    """
    if max_fill_gap_sec is not None:
        return float(max_fill_gap_sec)
    if filled_nan_log_sec is not None:
        return float(filled_nan_log_sec)
    return None


def count_long_filled_windows(filled_run_sec, starts, seq_len, threshold_sec):
    """Windows whose rows include a filled run longer than ``threshold_sec``.

    ``None`` threshold means the cutoff is unset. No run array means none
    of the windows contain a filled gap.
    """
    if threshold_sec is None:
        return None
    starts = [] if starts is None else list(starts)
    if filled_run_sec is None or not starts:
        return 0
    runs = np.asarray(filled_run_sec, dtype=np.float64).reshape(-1)
    if seq_len is None:
        length = int(runs.size)
    else:
        length = int(seq_len)
    if length < 1:
        return 0
    limit = float(threshold_sec)
    n_long = 0
    for start in starts:
        s = int(start)
        piece = runs[s : s + length]
        if piece.size and np.nanmax(piece) > limit:
            n_long += 1
    return n_long


def dataset_missing_rate(values, present=None) -> float:
    """Fraction of observation cells that are real missing data.

    A cell counts as missing when it is NaN, or when ``present`` is False
    on that row (even if the stored value is finite). ``+inf`` and ``-inf``
    are not missing. Callers pass the loaded observation values only:
    training-time masks and indicate 0/1 channels are not an input.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.size == 0:
        return 0.0
    missing = np.isnan(arr)
    if present is not None:
        flags = np.asarray(present, dtype=bool).reshape(-1)
        if flags.shape[0] != missing.shape[0]:
            raise ValueError(
                f"present length {flags.shape[0]} != series length {missing.shape[0]}"
            )
        absent = ~flags
        if missing.ndim == 1:
            missing = np.logical_or(missing, absent)
        else:
            shape = (-1,) + (1,) * (missing.ndim - 1)
            missing = np.logical_or(missing, absent.reshape(shape))
    return float(np.mean(missing))


def apply_nan_window_kwargs(dataset, kwargs, default_excess=0.2) -> None:
    """Store NaN-window options. A missing excess key uses ``default_excess``."""
    dataset.legacy_windowing = bool(
        kwargs.get("legacy_windowing", getattr(dataset, "legacy_windowing", False))
    )
    dataset.max_window_nan_ratio_off = bool(
        kwargs.get("max_window_nan_ratio_off", False)
    )
    if "max_window_nan_ratio" in kwargs:
        dataset.max_window_nan_ratio = kwargs["max_window_nan_ratio"]
    else:
        dataset.max_window_nan_ratio = None
    if "max_window_nan_excess" in kwargs:
        dataset.max_window_nan_excess = kwargs["max_window_nan_excess"]
    else:
        dataset.max_window_nan_excess = default_excess


def nan_window_threshold(dataset, minimum_nan_ratio=None):
    """Resolve the keep-threshold for one re-cut.

    Returns ``(threshold or None, label, dataset_missing_rate)``. ``None``
    means drop nothing: the ratio was turned off, legacy windowing left
    both rules off, the excess is unset, or the threshold is at least 1.
    An explicit ``minimum_nan_ratio`` argument, then an explicit
    ``max_window_nan_ratio``, are absolute caps. Otherwise the cap is
    ``dataset_missing_rate + max_window_nan_excess``.
    """
    rate = getattr(dataset, "dataset_missing_rate", None)
    if minimum_nan_ratio is not None:
        thr = float(minimum_nan_ratio)
        kind = "absolute"
    elif getattr(dataset, "max_window_nan_ratio_off", False):
        return None, "off", rate
    elif getattr(dataset, "max_window_nan_ratio", None) is not None:
        thr = float(dataset.max_window_nan_ratio)
        kind = "absolute"
    elif getattr(dataset, "legacy_windowing", False):
        return None, "off", rate
    else:
        excess = getattr(dataset, "max_window_nan_excess", None)
        if excess is None:
            return None, "off", rate
        base = 0.0 if rate is None else float(rate)
        thr = base + float(excess)
        kind = "relative"
    if thr >= 1.0:
        return None, f"{thr:.6g} ({kind}, >=1)", rate
    return thr, f"{thr:.6g} ({kind})", rate


def begin_window_recut(dataset) -> None:
    """Clear the per-re-cut window stats this ``update_sequence_length`` will fill."""
    dataset.window_recut_stats = {
        "windows_kept": 0,
        "windows_dropped_nan": 0,
        "windows_dropped_shift": 0,
        "gap_splits": 0,
        "mean_start_shift": 0.0,
        "dataset_missing_rate": getattr(dataset, "dataset_missing_rate", None),
        "nan_ratio_threshold": None,
        "nan_ratio_threshold_state": "na",
        "gap_quantile_len": getattr(dataset, "dataset_gap_quantile", None),
        "gap_split_threshold": None,
        "gap_split_threshold_state": "na",
        "block_mask_fraction": getattr(dataset, "block_mask_fraction", None),
    }


def _window_recut_stats(dataset) -> dict:
    stats = getattr(dataset, "window_recut_stats", None)
    if stats is None:
        begin_window_recut(dataset)
        stats = dataset.window_recut_stats
    return stats


def note_shift_stats(dataset, shift_stats) -> None:
    """Record unfit-start drops and the mean sample shift of kept windows."""
    stats = _window_recut_stats(dataset)
    if not shift_stats:
        stats["windows_dropped_shift"] = 0
        stats["mean_start_shift"] = 0.0
        return
    count = int(shift_stats.get("shift_count", 0))
    stats["windows_dropped_shift"] = int(shift_stats.get("dropped_unfit", 0))
    if count:
        stats["mean_start_shift"] = float(shift_stats.get("shift_sum", 0.0)) / count
    else:
        stats["mean_start_shift"] = 0.0


def snapshot_window_stats(dataset) -> dict:
    """Copy the latest re-cut stats. Missing datasets report zeros and ``na``."""
    raw = getattr(dataset, "window_recut_stats", None) or {}

    def _optional(key):
        if key not in raw or raw[key] is None:
            return None
        number = float(raw[key])
        if not np.isfinite(number):
            return None
        return number

    def _count(key):
        value = raw.get(key, 0)
        if value is None:
            return 0.0
        return float(value)

    return {
        "windows_kept": _count("windows_kept"),
        "windows_dropped_nan": _count("windows_dropped_nan"),
        "windows_dropped_shift": _count("windows_dropped_shift"),
        "gap_splits": _count("gap_splits"),
        "mean_start_shift": _count("mean_start_shift"),
        "dataset_missing_rate": _optional("dataset_missing_rate"),
        "nan_ratio_threshold": _optional("nan_ratio_threshold"),
        "nan_ratio_threshold_state": str(raw.get("nan_ratio_threshold_state", "na")),
        "gap_quantile_len": _optional("gap_quantile_len"),
        "gap_split_threshold": _optional("gap_split_threshold"),
        "gap_split_threshold_state": str(raw.get("gap_split_threshold_state", "na")),
        "block_mask_fraction": _optional("block_mask_fraction"),
    }


def log_nan_window_filter(dataset, threshold_label, dropped, rate) -> None:
    """One re-cut line: missing rate, threshold, and how many windows dropped."""
    name = type(dataset).__name__
    rate_s = "na" if rate is None else f"{float(rate):.6g}"
    print(
        f"[{name}] dataset_missing_rate={rate_s} "
        f"threshold={threshold_label} windows_dropped={int(dropped)}"
    )


def log_current_nan_window_rule(dataset, dropped, minimum_nan_ratio=None) -> None:
    threshold, label, rate = nan_window_threshold(dataset, minimum_nan_ratio)
    stats = _window_recut_stats(dataset)
    if rate is None:
        stats["dataset_missing_rate"] = None
    else:
        stats["dataset_missing_rate"] = float(rate)
    if threshold is None:
        stats["nan_ratio_threshold"] = None
        stats["nan_ratio_threshold_state"] = "off"
    else:
        stats["nan_ratio_threshold"] = float(threshold)
        stats["nan_ratio_threshold_state"] = "value"
    stats["windows_dropped_nan"] = int(dropped)
    log_nan_window_filter(dataset, label, dropped, rate)


def _seq_as_array(seq):
    if hasattr(seq, "detach"):
        return seq.detach().cpu().numpy()
    return np.asarray(seq)


def filter_window_starts(dataset, starts, minimum_nan_ratio=None):
    """Drop starts whose window NaN fraction is above the resolved threshold.

    The keep test is ``nan_fraction <= threshold`` in float64. A float16
    mean maps 4001/8000 onto 0.5, so a window just above a 0.5 cap would
    be kept. A threshold of None, or one that resolved to at least 1,
    keeps every start. Logs the dataset missing rate, the threshold, and
    the number dropped.
    """
    starts_arr = np.asarray([] if starts is None else list(starts), dtype=np.int64)
    threshold, label, rate = nan_window_threshold(dataset, minimum_nan_ratio)
    stats = _window_recut_stats(dataset)
    if rate is None:
        stats["dataset_missing_rate"] = None
    else:
        stats["dataset_missing_rate"] = float(rate)
    if threshold is None:
        stats["nan_ratio_threshold"] = None
        stats["nan_ratio_threshold_state"] = "off"
        stats["windows_dropped_nan"] = 0
        stats["windows_kept"] = int(starts_arr.size)
        log_nan_window_filter(dataset, label, 0, rate)
        return starts_arr
    seq = _seq_as_array(dataset.seq)
    length = int(dataset.seq_len)
    kept = []
    for idx in starts_arr:
        window = seq[int(idx) : int(idx) + length]
        if window.size == 0:
            continue
        nan_ratio = float(np.mean(np.isnan(window), dtype=np.float64))
        if nan_ratio <= float(threshold):
            kept.append(int(idx))
    dropped = int(starts_arr.size - len(kept))
    stats["nan_ratio_threshold"] = float(threshold)
    stats["nan_ratio_threshold_state"] = "value"
    stats["windows_dropped_nan"] = dropped
    stats["windows_kept"] = int(len(kept))
    log_nan_window_filter(dataset, label, dropped, rate)
    return np.asarray(kept, dtype=np.int64)


def log_window_recut(dataset) -> None:
    """Print window count and how many hold a too-long filled NaN run."""
    threshold = recut_threshold_sec(
        getattr(dataset, "max_fill_gap_sec", None),
        getattr(dataset, "filled_nan_log_sec", None),
    )
    starts = getattr(dataset, "data_idx", None)
    seq_len = getattr(dataset, "seq_len", None)
    n_windows = 0 if starts is None else len(starts)
    n_long = count_long_filled_windows(
        getattr(dataset, "filled_run_sec", None),
        starts,
        seq_len if seq_len is not None else _seq_rows(dataset),
        threshold,
    )
    if seq_len is None:
        seq_s = "None"
    else:
        seq_s = str(int(seq_len))
    if n_long is None:
        long_s = "na"
        thr_s = "unset"
    else:
        long_s = str(int(n_long))
        thr_s = f"{float(threshold):.6g}"
    name = type(dataset).__name__
    print(
        f"[{name}] re-cut seq_len={seq_s} windows={n_windows} "
        f"long_filled_windows={long_s} threshold_sec={thr_s}"
    )


def missing_row_mask(values, present=None) -> np.ndarray:
    """True on rows that are real missing data.

    A row is missing when any observation cell is NaN, or when ``present``
    is False (even if the stored values are finite). ``+inf`` and ``-inf``
    are not missing. Training-time masks are not an input.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    missing = np.isnan(arr).any(axis=1)
    if present is not None:
        flags = np.asarray(present, dtype=bool).reshape(-1)
        if flags.shape[0] != missing.shape[0]:
            raise ValueError(
                f"present length {flags.shape[0]} != series length {missing.shape[0]}"
            )
        missing = np.logical_or(missing, ~flags)
    return missing


def contiguous_run_lengths(mask) -> np.ndarray:
    """Lengths of contiguous True runs. Empty when nothing is missing."""
    flags = np.asarray(mask, dtype=bool).reshape(-1)
    if flags.size == 0 or not flags.any():
        return np.asarray([], dtype=np.int64)
    padded = np.concatenate([[False], flags, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return (ends - starts).astype(np.int64)


def gap_length_quantile(lengths, quantile) -> float | None:
    """Linear quantile of missing-run lengths. None when there are no runs."""
    if quantile is None:
        return None
    runs = np.asarray(lengths, dtype=np.float64).reshape(-1)
    if runs.size == 0:
        return None
    return float(np.quantile(runs, float(quantile)))


def gap_split_limit(seq_len, frac, gap_quantile):
    """Logged sample threshold for one re-cut, or None when the rule is off.

    The rule is off when ``frac`` is unset. Otherwise the logged threshold
    is ``frac * seq_len``, raised to ``gap_quantile`` when that length is
    set. Whether a run actually splits is :func:`run_exceeds_gap_split`:
    a run longer than the fraction still splits when it equals the
    quantile, so the logged number is not a strict ``>`` cutoff.
    """
    if frac is None or seq_len is None:
        return None
    limit = float(frac) * int(seq_len)
    if gap_quantile is not None:
        limit = max(limit, float(gap_quantile))
    return limit


def run_exceeds_gap_split(length, seq_len, frac, gap_quantile) -> bool:
    """True when this missing run is cut out of windows.

    The floor is ``frac * seq_len``. A run that is not strictly longer
    than that floor stays, even if it is the quantile. A run that is
    strictly longer than the floor splits when the quantile is unset or
    no higher than the floor. When the quantile is above the floor, the
    run splits if it is at least that quantile (``>=``, not ``>``).

    A lone run, or several runs of the same length, is its own 0.99
    quantile. A strict ``>`` against ``max(floor, quantile)`` would
    never cut it. ``>=`` against that quantile keeps the floor as a
    strict lower bound and still cuts a run that only ties the quantile
    once the run is longer than a quarter of the window. A run that is
    longer than the floor but shorter than a higher quantile of other
    runs stays: the quantile is still the upper bar, it just cannot
    hide a run that is the bar.
    """
    if frac is None or seq_len is None:
        return False
    floor = float(frac) * int(seq_len)
    length = int(length)
    if gap_quantile is None or float(gap_quantile) <= floor:
        return length > floor
    return length >= float(gap_quantile)


def filled_gap_frac_cuts(
    spans, seq_len, frac, num_frames, gap_quantile=None
) -> np.ndarray:
    """Cuts that isolate missing runs the gap-split rule removes.

    ``frac`` unset returns no cuts. See :func:`run_exceeds_gap_split`
    for the comparison. A qualifying run is cut at its first row and at
    the first row after it, same as ``max_fill_gap_sec``.
    """
    if gap_split_limit(seq_len, frac, gap_quantile) is None:
        return np.asarray([], dtype=np.int64)
    n = int(num_frames)
    cuts = []
    for span in spans or []:
        rows = np.asarray(span, dtype=np.int64).ravel()
        if rows.size == 0 or not run_exceeds_gap_split(
            int(rows.size), seq_len, frac, gap_quantile
        ):
            continue
        start = int(rows.min())
        after = int(rows.max()) + 1
        if 0 < start < n:
            cuts.append(start)
        if 0 < after < n:
            cuts.append(after)
    if not cuts:
        return np.asarray([], dtype=np.int64)
    return np.unique(np.asarray(cuts, dtype=np.int64))


def log_gap_split_recut(dataset, seq_len, num_frames) -> None:
    """Log the quantile gap length, the effective threshold, and split count."""
    frac = getattr(dataset, "max_gap_window_frac", None)
    qlen = getattr(dataset, "dataset_gap_quantile", None)
    name = type(dataset).__name__
    q_s = "na" if qlen is None else f"{float(qlen):.6g}"
    limit = gap_split_limit(seq_len, frac, qlen)
    stats = _window_recut_stats(dataset)
    stats["gap_quantile_len"] = None if qlen is None else float(qlen)
    if limit is None:
        stats["gap_split_threshold"] = None
        stats["gap_split_threshold_state"] = "off"
        stats["gap_splits"] = 0
        print(
            f"[{name}] gap_quantile_len={q_s} "
            f"gap_split_threshold=off gap_splits=0"
        )
        return
    spans = getattr(dataset, "missing_run_spans", None)
    if spans is None:
        spans = getattr(dataset, "filled_gap_spans", None)
    cuts = filled_gap_frac_cuts(
        spans, seq_len, frac, num_frames, gap_quantile=qlen
    )
    stats["gap_split_threshold"] = float(limit)
    stats["gap_split_threshold_state"] = "value"
    stats["gap_splits"] = int(cuts.size)
    print(
        f"[{name}] gap_quantile_len={q_s} "
        f"gap_split_threshold={float(limit):.6g} gap_splits={int(cuts.size)}"
    )


def combine_boundaries(base, extra, num_frames):
    """Union of split indices strictly inside the series. None if there are none."""
    parts = []
    if base is not None:
        parts.append(np.asarray(base, dtype=np.int64).ravel())
    if extra is not None:
        extra_arr = np.asarray(extra, dtype=np.int64).ravel()
        if extra_arr.size:
            parts.append(extra_arr)
    if not parts:
        return None
    return _boundary_cuts(np.concatenate(parts), num_frames)


def resolve_recut_length(
    num_frames, requested, base_boundaries, spans, frac, gap_quantile=None
):
    """Sequence length and the split indices for this re-cut.

    ``max_gap_window_frac`` depends on the length, so a cap that shortens
    the window is applied again until the cuts stop changing. No cuts
    leaves the requested length alone. ``gap_quantile`` is the stored
    missing-run quantile, in samples. None leaves the fraction alone.
    """
    n = int(num_frames)
    seq_len = int(requested)
    seen = set()
    boundaries = combine_boundaries(base_boundaries, None, n)
    while seq_len not in seen:
        seen.add(seq_len)
        extra = filled_gap_frac_cuts(
            spans, seq_len, frac, n, gap_quantile=gap_quantile
        )
        boundaries = combine_boundaries(base_boundaries, extra, n)
        if boundaries is None:
            return seq_len, None
        span = longest_unbroken_span(n, boundaries)
        if span >= 1 and seq_len > span:
            seq_len = span
            continue
        return seq_len, boundaries
    return seq_len, boundaries


def groups_from_boundaries(num_frames, boundaries) -> np.ndarray:
    """Contiguous ids. A new id starts at each split index."""
    n = int(num_frames)
    group = np.zeros(n, dtype=np.int64)
    cuts = _boundary_cuts(boundaries, n)
    if cuts is None:
        return group
    change = np.zeros(n, dtype=bool)
    change[cuts] = True
    gid = 0
    for i in range(1, n):
        if change[i]:
            gid += 1
        group[i] = gid
    return group


def observed_row_mask(seq, observed_per_row):
    """True where a window is allowed to start.

    ``observed_per_row`` is the aligned ``present`` flag when the file has
    one. Otherwise a row is observed when it is not entirely NaN.
    """
    if observed_per_row is not None:
        return np.asarray(observed_per_row, dtype=bool).reshape(-1)
    arr = np.asarray(seq.detach().cpu().numpy() if hasattr(seq, "detach") else seq)
    if arr.ndim <= 1:
        return np.isfinite(np.reshape(arr, -1))
    finite = np.isfinite(np.reshape(arr, (arr.shape[0], -1)))
    return finite.any(axis=1)


def missing_run_spans(mask) -> list:
    """Index arrays of contiguous missing rows. Empty when nothing is missing."""
    flags = np.asarray(mask, dtype=bool).reshape(-1)
    if flags.size == 0 or not flags.any():
        return []
    padded = np.concatenate([[False], flags, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [np.arange(int(start), int(end)) for start, end in zip(starts, ends)]


def install_simulated_window_rules(dataset, kwargs) -> None:
    """Give a Lorenz or SHO series the same window rules as Xhro.

    Blank keys take the Xhro defaults. ``legacy_defaults`` or
    ``legacy_windowing`` true restores historical placement: starts may
    land on a missing row, ``find_indices`` is not strict, and the
    fraction split and the NaN filter stay off unless a key is set.
    Call this after ``dataset.seq`` exists, then
    :func:`refresh_simulated_gap_stats`.
    """
    from dvae.dataset.xhro_proper_dataset import resolve_windowing

    windowing = resolve_windowing(
        legacy_windowing=kwargs.get("legacy_windowing", kwargs.get("legacy_defaults")),
        legacy_defaults=kwargs.get("legacy_defaults"),
        start_on_observed=kwargs.get("start_on_observed"),
        strict_non_overlap=kwargs.get("strict_non_overlap"),
        max_window_nan_ratio=kwargs.get("max_window_nan_ratio"),
        max_window_nan_excess=kwargs.get("max_window_nan_excess"),
        max_gap_window_frac=kwargs.get("max_gap_window_frac"),
        gap_split_quantile=kwargs.get("gap_split_quantile"),
    )
    if kwargs.get("max_window_nan_ratio_off"):
        windowing["max_window_nan_ratio_off"] = True
        windowing["max_window_nan_ratio"] = None
    for key, value in windowing.items():
        setattr(dataset, key, value)


def refresh_simulated_gap_stats(dataset) -> None:
    """Missing rate, run spans, and the gap quantile of the loaded series.

    A damped SHO stores one span list per instance. Every other series
    stores one list of row spans. The quantile pools every run.
    """
    arr = _seq_as_array(dataset.seq)
    dataset.dataset_missing_rate = float(dataset_missing_rate(arr))
    if getattr(dataset, "is_damped_sho", False) and getattr(arr, "ndim", 1) == 2:
        pieces = [arr[:, index] for index in range(arr.shape[1])]
    else:
        pieces = [arr]
    per_spans = []
    length_parts = []
    for piece in pieces:
        mask = missing_row_mask(piece)
        per_spans.append(missing_run_spans(mask))
        length_parts.append(contiguous_run_lengths(mask))
    lengths = (
        np.concatenate(length_parts)
        if length_parts
        else np.asarray([], dtype=np.int64)
    )
    quantile = getattr(dataset, "gap_split_quantile", None)
    if quantile is None:
        dataset.dataset_gap_quantile = None
    else:
        dataset.dataset_gap_quantile = gap_length_quantile(lengths, quantile)
    if len(per_spans) == 1:
        dataset.missing_run_spans = per_spans[0]
        dataset.instance_gap_spans = None
    else:
        dataset.missing_run_spans = []
        dataset.instance_gap_spans = per_spans


def recut_flat_windows(dataset, new_seq_len, minimum_nan_ratio=None):
    """Window starts for the series currently on ``dataset.seq``.

    Same placement as ``Xhro.update_sequence_length``: gap cuts, an
    observed start, then the relative NaN filter. Returns the starts
    after that filter. ``None`` means ``seq_len`` was unset and the
    caller keeps one placeholder window. Does not assign ``data_idx``.
    """
    from dvae.dataset.utils import data_utils

    begin_window_recut(dataset)
    seq = _seq_as_array(dataset.seq)
    num_frames = int(seq.shape[0])
    if new_seq_len is None:
        log_gap_split_recut(dataset, None, num_frames)
        log_current_nan_window_rule(dataset, 0, minimum_nan_ratio)
        stats = _window_recut_stats(dataset)
        stats["windows_kept"] = 1
        return None
    requested = int(new_seq_len)
    spans = getattr(dataset, "missing_run_spans", None)
    seq_len, boundaries = resolve_recut_length(
        num_frames,
        requested,
        getattr(dataset, "split_boundaries", None),
        spans,
        getattr(dataset, "max_gap_window_frac", None),
        gap_quantile=getattr(dataset, "dataset_gap_quantile", None),
    )
    if boundaries is not None and seq_len < requested:
        print(
            f"[{type(dataset).__name__}] seq_len {requested} exceeds the "
            f"longest unbroken span {seq_len}; using {seq_len}."
        )
    dataset.seq_len = seq_len
    log_gap_split_recut(dataset, dataset.seq_len, num_frames)
    strict = bool(getattr(dataset, "strict_non_overlap", False))
    shift_stats = {}
    observed = None
    if strict and getattr(dataset, "start_on_observed", False):
        observed = observed_row_mask(
            dataset.seq, getattr(dataset, "observed_per_row", None)
        )
    count = num_frames // dataset.seq_len if dataset.seq_len else 0
    all_indices = data_utils.find_indices(
        num_frames,
        dataset.seq_len,
        count,
        boundaries=boundaries,
        strict_non_overlap=strict,
        observed=observed,
        shift_stats=shift_stats if observed is not None else None,
    )
    if observed is not None:
        # Strict mode already shifted inside find_indices. Shift again
        # without adding to the stats, matching Xhro.
        all_indices = shift_starts_onto_observed(
            all_indices,
            observed,
            dataset.seq_len,
            boundaries,
            num_frames,
        )
    elif getattr(dataset, "start_on_observed", False):
        observed = observed_row_mask(
            dataset.seq, getattr(dataset, "observed_per_row", None)
        )
        all_indices = shift_starts_onto_observed(
            all_indices,
            observed,
            dataset.seq_len,
            boundaries,
            num_frames,
            shift_stats=shift_stats,
        )
    else:
        n = 0 if all_indices is None else int(np.asarray(all_indices).size)
        shift_stats["dropped_unfit"] = int(shift_stats.get("dropped_unfit", 0))
        shift_stats["shift_sum"] = float(shift_stats.get("shift_sum", 0.0))
        shift_stats["shift_count"] = int(shift_stats.get("shift_count", 0)) + n
    note_shift_stats(dataset, shift_stats)
    return filter_window_starts(dataset, all_indices, minimum_nan_ratio)


def _piece_containing(start: int, cuts, num_frames: int):
    left = 0
    right = int(num_frames)
    if cuts is None:
        return left, right
    for cut in cuts:
        cut = int(cut)
        if cut <= start:
            left = cut
        else:
            right = cut
            break
    return left, right


def shift_starts_onto_observed(
    starts, observed, seq_len, boundaries, num_frames, shift_stats=None
):
    """Move a start that lands on a missing sample to the next observed row.

    The window must still lie inside the piece that contains the original
    start. A start that cannot move without leaving that piece is dropped.
    An already observed start is kept. Duplicate landing rows are kept once.

    ``shift_stats``, when passed, receives ``dropped_unfit`` (starts that
    could not fit in the piece), ``shift_sum`` and ``shift_count`` for the
    kept landings (shift of 0 included).
    """
    n = int(num_frames)
    length = int(seq_len)
    flags = np.asarray(observed, dtype=bool).reshape(-1)
    if flags.shape[0] != n:
        raise ValueError(
            f"observed mask length {flags.shape[0]} != series length {n}"
        )
    cuts = _boundary_cuts(boundaries, n)
    kept = []
    seen = set()
    dropped_unfit = 0
    shift_sum = 0.0
    shift_count = 0
    for raw in np.asarray([] if starts is None else starts).ravel():
        start = int(raw)
        if start < 0 or start >= n:
            continue
        left, right = _piece_containing(start, cuts, n)
        if flags[start]:
            landing = start
        else:
            landing = start + 1
            while landing < right and not flags[landing]:
                landing += 1
        if landing >= right or landing + length > right:
            dropped_unfit += 1
            continue
        if landing in seen:
            continue
        seen.add(landing)
        shift_sum += float(landing - start)
        shift_count += 1
        kept.append(landing)
    if shift_stats is not None:
        shift_stats["dropped_unfit"] = (
            int(shift_stats.get("dropped_unfit", 0)) + dropped_unfit
        )
        shift_stats["shift_sum"] = float(shift_stats.get("shift_sum", 0.0)) + shift_sum
        shift_stats["shift_count"] = int(shift_stats.get("shift_count", 0)) + shift_count
    return np.asarray(kept, dtype=np.int64)


def _seq_rows(dataset) -> int:
    seq = getattr(dataset, "seq", None)
    if seq is None:
        return 0
    return int(len(seq))


def _median_positive_dt(t) -> float:
    if t is None:
        return 0.0
    tt = np.asarray(t, dtype=np.float64).reshape(-1)
    if tt.size < 2:
        return 0.0
    step = np.diff(tt)
    step = step[np.isfinite(step) & (step > 0)]
    if step.size == 0:
        return 0.0
    return float(np.median(step))


def _combine_filled_run(df: pd.DataFrame) -> pd.DataFrame:
    x, y = f"{FILLED_RUN_COL}_x", f"{FILLED_RUN_COL}_y"
    if x not in df.columns and y not in df.columns:
        return df
    def _side(column: str) -> np.ndarray:
        if column not in df.columns:
            return np.zeros(len(df), dtype=np.float64)
        return np.asarray(
            pd.to_numeric(df[column], errors="coerce").fillna(0.0),
            dtype=np.float64,
        )
    df[FILLED_RUN_COL] = np.maximum(_side(x), _side(y))
    return df.drop(columns=[c for c in (x, y) if c in df.columns])


def window_group_ids(n, segment_id, hard_indices, filled_spans) -> np.ndarray:
    """Contiguous ids. A window may not contain two ids.

    ``filled_spans`` are row-index arrays of filled gaps that must be
    isolated (the rows themselves, then the samples after them).
    """
    n = int(n)
    cuts = np.zeros(n, dtype=bool)
    if segment_id is not None and n > 1:
        sid = np.asarray(segment_id)
        cuts[1:] |= sid[1:] != sid[:-1]
    for raw in np.asarray(hard_indices, dtype=np.int64).ravel():
        gi = int(raw)
        if 0 < gi < n:
            cuts[gi] = True
    for span in filled_spans:
        rows = np.asarray(span, dtype=np.int64).ravel()
        if rows.size == 0:
            continue
        start = int(rows[0])
        after = int(rows[-1]) + 1
        if 0 < start < n:
            cuts[start] = True
        if 0 < after < n:
            cuts[after] = True
    group = np.zeros(n, dtype=np.int64)
    gid = 0
    for i in range(1, n):
        if cuts[i]:
            gid += 1
        group[i] = gid
    return group


def _blank_absent_rows(df: pd.DataFrame, present: np.ndarray) -> None:
    absent = ~np.asarray(present, dtype=bool)
    if not absent.any():
        return
    for col in list(df.columns):
        if col == "datetime" or str(col).startswith("__"):
            continue
        values = np.asarray(df[col], dtype=np.float64).copy()
        if values.shape != absent.shape:
            continue
        values[absent] = np.nan
        df[col] = values


def _read_present(npz, n: int):
    if "present" not in npz.files:
        return None
    present = np.asarray(npz["present"])
    if present.ndim == 2 and 1 in present.shape:
        present = present.reshape(-1)
    if present.shape != (n,):
        raise ValueError(
            f"present length {present.shape} != series length {n}"
        )
    return _as_bool(present)


def _read_segment_id(npz, n: int):
    if "segment_id" in npz.files:
        sid = np.asarray(npz["segment_id"])
        if sid.ndim == 2 and 1 in sid.shape:
            sid = sid.reshape(-1)
        if sid.shape != (n,):
            raise ValueError(
                f"segment_id length {sid.shape} != series length {n}"
            )
        if sid.dtype.kind == "f" and not np.all(np.isfinite(sid)):
            raise ValueError("segment_id contains non-finite values")
        return sid.astype(np.int64)
    if "segment_bounds" in npz.files:
        return _segment_id_from_bounds(npz["segment_bounds"], n)
    return None


def _segment_id_from_bounds(bounds, n: int) -> np.ndarray:
    pairs = _half_open_pairs(bounds, n)
    sid = np.full(n, -1, dtype=np.int64)
    for i, (start, end) in enumerate(pairs):
        a = int(start)
        b = int(end)
        if b < a:
            raise ValueError(
                f"segment_bounds pair {(a, b)} has end < start"
            )
        lo = min(max(a, 0), n)
        hi = min(max(b, 0), n)
        sid[lo:hi] = i
    if np.any(sid < 0):
        warnings.warn(
            "segment_bounds did not cover every row; uncovered runs "
            "become their own window groups.",
            UserWarning,
            stacklevel=2,
        )
        next_id = int(sid.max()) + 1 if np.any(sid >= 0) else 0
        i = 0
        while i < n:
            if sid[i] >= 0:
                i += 1
                continue
            j = i + 1
            while j < n and sid[j] < 0:
                j += 1
            sid[i:j] = next_id
            next_id += 1
            i = j
    return sid


def _half_open_pairs(bounds, n: int) -> np.ndarray:
    """Return ``(S, 2)`` half-open pairs.

    A 1-d array is the edges. A ``(S, 2)`` array whose ends meet the next
    start, or whose largest end equals ``n``, is already half-open.
    Otherwise ends are treated as inclusive.
    """
    raw = np.asarray(bounds)
    if raw.ndim == 1:
        edges = raw.astype(np.int64).ravel()
        if edges.size < 2:
            raise ValueError(
                f"segment_bounds edge array is too short: shape {raw.shape}"
            )
        return np.column_stack([edges[:-1], edges[1:]])
    if raw.ndim != 2:
        raise ValueError(f"segment_bounds shape {raw.shape} is not 1-d or 2-d")
    if raw.shape[1] == 2:
        pairs = raw.astype(np.int64)
    elif raw.shape[0] == 2:
        pairs = raw.T.astype(np.int64)
    else:
        raise ValueError(
            f"segment_bounds shape {raw.shape} is not (S, 2) or (2, S)"
        )
    if pairs.size == 0:
        return pairs.reshape(0, 2)
    ends = pairs[:, 1]
    starts = pairs[:, 0]
    meets = pairs.shape[0] > 1 and bool(np.all(ends[:-1] == starts[1:]))
    if meets or int(np.max(ends)) == n:
        return pairs
    inclusive = pairs.copy()
    inclusive[:, 1] = inclusive[:, 1] + 1
    return inclusive


def _read_gap_indices(npz) -> np.ndarray:
    if "gap_indices" not in npz.files:
        return np.asarray([], dtype=np.int64)
    return np.asarray(npz["gap_indices"], dtype=np.int64).ravel()


def _long_filled_spans(npz, n, t, present, max_fill_gap_sec):
    if max_fill_gap_sec is None:
        return []
    limit = float(max_fill_gap_sec)
    table = _gap_table(npz)
    if table is None or "gap_duration_s" not in table or "gap_filled" not in table:
        warnings.warn(
            "max_fill_gap_sec is set but the NPZ has no gap_duration_s/"
            "gap_filled table; filled gaps are not extra split points.",
            UserWarning,
            stacklevel=2,
        )
        return []
    duration = np.asarray(table["gap_duration_s"], dtype=np.float64).reshape(-1)
    filled = _as_bool(table["gap_filled"]).reshape(-1)
    n_missing = table.get("gap_n_missing")
    gap_index = table.get("gap_index")
    gap_t = table.get("gap_t_start")
    spans = []
    for i, (is_filled, dur) in enumerate(zip(filled, duration)):
        if not is_filled:
            continue
        if not np.isfinite(dur) or dur <= limit:
            continue
        start = _locate_gap_start(
            n,
            t,
            None if gap_index is None else gap_index[i],
            None if gap_t is None else gap_t[i],
        )
        count = None
        if n_missing is not None and np.isfinite(n_missing[i]):
            count = int(n_missing[i])
        rows = _filled_rows(n, present, start, count)
        if rows.size == 0:
            warnings.warn(
                f"filled gap {i} (duration_s={dur}) could not be placed "
                "on the series; it is not a split point.",
                UserWarning,
                stacklevel=2,
            )
            continue
        spans.append(rows)
    return spans


def _gap_table(npz):
    cols = {}
    for key in _TABLE_KEYS:
        if key in npz.files:
            cols[key] = np.atleast_1d(np.asarray(npz[key]))
    if not cols:
        return None
    lengths = {key: int(np.asarray(value).reshape(-1).shape[0]) for key, value in cols.items()}
    # gap_kind may be a 0-d label accidentally; a real table shares one length.
    unique = set(lengths.values())
    if len(unique) != 1:
        raise ValueError(
            "Gap-table columns must share one length, got "
            + ", ".join(f"{key}={length}" for key, length in sorted(lengths.items()))
        )
    return cols


def _locate_gap_start(n, t, gap_index, gap_t_start):
    if gap_index is not None and np.isfinite(gap_index):
        gi = int(gap_index)
        if 0 <= gi < n:
            return gi
    if (
        t is not None
        and gap_t_start is not None
        and np.isfinite(gap_t_start)
    ):
        tt = np.asarray(t, dtype=np.float64).reshape(-1)
        if tt.shape[0] != n:
            return None
        pos = int(np.argmin(np.abs(tt - float(gap_t_start))))
        tol = _half_sample(tt)
        if abs(float(tt[pos]) - float(gap_t_start)) <= tol:
            return pos
    return None


def _half_sample(t: np.ndarray) -> float:
    if t.size < 2:
        return 0.0
    step = np.diff(t)
    step = step[np.isfinite(step) & (step > 0)]
    if step.size == 0:
        return 0.0
    return 0.5 * float(np.median(step))


def _filled_rows(n, present, start, n_missing) -> np.ndarray:
    if start is None:
        return np.asarray([], dtype=np.int64)
    if n_missing is not None and n_missing > 0:
        end = min(n, start + int(n_missing))
        return np.arange(start, end, dtype=np.int64)
    if present is not None:
        end = start
        flags = np.asarray(present, dtype=bool)
        while end < n and not flags[end]:
            end += 1
        if end == start:
            return np.asarray([], dtype=np.int64)
        return np.arange(start, end, dtype=np.int64)
    return np.asarray([start], dtype=np.int64)


def _as_bool(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == bool:
        return arr
    if arr.dtype.kind in "SUO":
        text = np.char.lower(np.char.strip(arr.astype(str)))
        return np.isin(text, ("1", "true", "yes", "y"))
    return np.asarray(arr, dtype=np.float64) != 0


def _bool_side(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.ones(len(df), dtype=bool)
    values = df[column]
    # Outer-join holes are not present.
    if values.dtype != bool:
        mask = values.isna()
        out = _as_bool(values.fillna(0))
        out = np.asarray(out, dtype=bool).copy()
        out[np.asarray(mask, dtype=bool)] = False
        return out
    # Boolean NA is possible after an outer merge of bool columns.
    filled = values.astype("boolean").fillna(False)
    return filled.to_numpy(dtype=bool)


def _factorize_pairs(left: pd.Series, right: pd.Series) -> np.ndarray:
    def _token(value):
        if pd.isna(value):
            return ""
        return str(int(value))

    key = [f"{_token(a)}|{_token(b)}" for a, b in zip(left, right)]
    codes, _ = pd.factorize(key, sort=False)
    return np.asarray(codes, dtype=np.int64)
