"""Pure selection of an evaluation window length."""

from __future__ import annotations

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
