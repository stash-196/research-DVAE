"""Pure selection of an evaluation window length."""

from __future__ import annotations

import numpy as np

from dvae.eval.eval_signal import fit_metrics_window, metrics_shrink_allowed
from dvae.eval.window_length import (
    METRICS_WINDOW_FLOOR,
    block_len_after_autofit,
    choose_metrics_window_length,
)


def test_request_with_windows_does_not_shrink():
    seen = []

    def counts_at(length):
        seen.append(length)
        if length == 20:
            return [1, 0]
        return [1, 1]

    choice = choose_metrics_window_length(20, counts_at)
    assert choice["length"] == 20
    assert choice["shrunk"] is False
    assert choice["reason"] is None
    assert seen == [20]


def test_longest_length_that_covers_every_subject():
    def counts_at(length):
        if length >= 12:
            return [0, 0]
        if length >= 9:
            return [1, 0]
        return [1, 1]

    choice = choose_metrics_window_length(20, counts_at, floor=4)
    assert choice["length"] == 8
    assert choice["shrunk"] is True
    assert choice["counts"] == [1, 1]
    assert choice["from_length"] == 20
    assert choice["subjects_covered"] == 2


def test_partial_coverage_keeps_the_longer_length():
    def counts_at(length):
        if length in (10, 4):
            return [1, 0]
        return [0, 0]

    choice = choose_metrics_window_length(20, counts_at, floor=4)
    assert choice["length"] == 10
    assert choice["subjects_covered"] == 1
    assert choice["shrunk"] is True


def test_explicit_length_does_not_probe_shorter_windows():
    seen = []

    def counts_at(length):
        seen.append(length)
        return [0, 0]

    choice = choose_metrics_window_length(20, counts_at, allow_shrink=False)
    assert choice["length"] is None
    assert choice["reason"] == "no_windows"
    assert choice["shrunk"] is False
    assert seen == [20]


def test_nothing_at_or_above_the_floor_is_skipped():
    def counts_at(length):
        return [0]

    choice = choose_metrics_window_length(6, counts_at, floor=METRICS_WINDOW_FLOOR)
    assert choice["length"] is None
    assert choice["reason"] == "no_windows"


def test_block_length_halves_only_after_a_shrink():
    assert block_len_after_autofit(30, 1000, shrunk=False) == 1000
    assert block_len_after_autofit(30, 1000, shrunk=True) == 15
    assert block_len_after_autofit(1, 1000, shrunk=True) == 1


def test_stored_shorter_than_requested_is_a_shrink_when_windows_exist():
    seen = []

    def counts_at(length):
        seen.append(length)
        return (10, [2, 0])

    choice = choose_metrics_window_length(30, counts_at)
    assert seen == [30]
    assert choice["length"] == 10
    assert choice["stored_length"] == 10
    assert choice["shrunk"] is True
    assert block_len_after_autofit(choice["length"], 1000, shrunk=True) == 5


def test_explicit_length_records_the_stored_cap_without_halving_the_block():
    def counts_at(length):
        return (10, [2, 0])

    choice = choose_metrics_window_length(30, counts_at, allow_shrink=False)
    assert choice["length"] == 10
    assert choice["stored_length"] == 10
    assert choice["shrunk"] is False
    assert block_len_after_autofit(choice["length"], 1000, shrunk=False) == 1000


def test_capped_empty_probe_is_not_repeated_above_the_group_span():
    seen = []

    def counts_at(length):
        seen.append(length)
        if length > 12:
            return (12, [0, 0])
        if length >= 9:
            return (length, [1, 0])
        return (length, [1, 1])

    choice = choose_metrics_window_length(100, counts_at, max_length=40, floor=4)
    assert seen[0] == 100
    assert all(length <= 12 for length in seen[1:])
    assert 40 not in seen
    assert choice["length"] == 8
    assert choice["shrunk"] is True
    assert choice["counts"] == [1, 1]


class _CappedWindows:
    """Cutter stores ``cap`` and still returns a window."""

    def __init__(self, n_rows, cap):
        self.seq = np.zeros((n_rows, 1), dtype=np.float64)
        self.seq_len = 4
        self.data_idx = [0]
        self.calls = []
        self.cap = int(cap)

    def update_sequence_length(self, new_seq_len=None, **kwargs):
        new = int(new_seq_len)
        self.calls.append(new)
        self.seq_len = min(new, self.cap)
        self.data_idx = [0]


class _RaiseAbove:
    """Lengths above ``cap`` raise and do not change ``seq_len``."""

    def __init__(self, n_rows, cap):
        self.seq = np.zeros((n_rows, 1), dtype=np.float64)
        self.seq_len = 4
        self.data_idx = [0]
        self.calls = []
        self.cap = int(cap)

    def update_sequence_length(self, new_seq_len=None, **kwargs):
        new = int(new_seq_len)
        self.calls.append(new)
        if new > self.cap:
            raise ValueError("no intra-stay window")
        self.seq_len = new
        self.data_idx = [0]


class _CapThenRaise:
    """Stores the cap, then raises. Shorter lengths still have a window."""

    def __init__(self, n_rows, cap):
        self.seq = np.zeros((n_rows, 1), dtype=np.float64)
        self.seq_len = 4
        self.data_idx = []
        self.calls = []
        self.cap = int(cap)

    def update_sequence_length(self, new_seq_len=None, **kwargs):
        new = int(new_seq_len)
        self.calls.append(new)
        if new > self.cap:
            self.seq_len = self.cap
            self.data_idx = []
            raise ValueError("no intra-stay window")
        self.seq_len = new
        self.data_idx = [0]


def test_metrics_shrink_allowed_follows_the_explicit_cli_keys():
    assert metrics_shrink_allowed(None) is True
    assert metrics_shrink_allowed({}) is True
    assert metrics_shrink_allowed({"auto_eval_block_len": None}) is True
    assert metrics_shrink_allowed({"auto_eval_block_len": 0}) is True
    assert metrics_shrink_allowed({"auto_eval_block_len": 1000}) is False
    assert metrics_shrink_allowed({"metrics_seq_len": 30}) is False


def test_capped_cutter_with_windows_shrinks_to_the_stored_length():
    dataset = _CappedWindows(40, 10)
    choice = fit_metrics_window(dataset, 30, {})
    assert dataset.calls == [30]
    assert choice["length"] == 10
    assert choice["stored_length"] == 10
    assert choice["shrunk"] is True
    assert block_len_after_autofit(choice["length"], 1000, shrunk=True) == 5


def test_explicit_block_disables_shrinking_but_records_the_stored_length():
    dataset = _CappedWindows(40, 10)
    choice = fit_metrics_window(dataset, 30, {"auto_eval_block_len": 1000})
    assert dataset.calls == [30]
    assert choice["length"] == 10
    assert choice["stored_length"] == 10
    assert choice["shrunk"] is False
    assert block_len_after_autofit(choice["length"], 1000, shrunk=False) == 1000

    other = _CappedWindows(40, 10)
    by_metrics = fit_metrics_window(other, 30, {"metrics_seq_len": 30})
    assert by_metrics["shrunk"] is False
    assert by_metrics["length"] == 10


def test_raising_cutter_counts_as_zero_windows_and_keeps_scanning():
    dataset = _RaiseAbove(10, 8)
    choice = fit_metrics_window(dataset, 20, {})
    assert dataset.calls[0] == 20
    assert 8 in dataset.calls
    assert len(dataset.calls) < 20
    assert choice["length"] == 8
    assert choice["shrunk"] is True


def test_cap_then_raise_does_not_reprobe_the_capped_range():
    dataset = _CapThenRaise(40, 12)
    choice = fit_metrics_window(dataset, 100, {})
    assert dataset.calls == [100, 11]
    assert choice["length"] == 11
    assert choice["shrunk"] is True
    assert choice["counts"] == [1]
