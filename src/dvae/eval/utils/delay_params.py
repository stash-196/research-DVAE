"""Hardcoded delay-embedding (tau, m) defaults.

Kept separate from benchmark construction so eval can resolve these values
without importing torch. ``benchmark_signals._get_delay_params`` re-exports
this function.
"""

from typing import Tuple


def _get_delay_params(dataset_name: str) -> Tuple[int, int]:
    """Return ``(time_delay, delay_dims)`` for a dataset name.

    Lorenz63 uses tau=10, m=3. Every other dataset uses tau=5, m=3.
    """
    if dataset_name == "Lorenz63":
        return 10, 3
    return 5, 3
