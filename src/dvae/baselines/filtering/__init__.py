"""Filtering imputation baselines (linear Kalman EM, Lorenz oracle, Tresp).

These modules are standalone. They are not wired into ``learning_algo`` or the
signal eval loop.
"""

from dvae.baselines.filtering.kalman_em import LinearGaussianEM

__all__ = ["LinearGaussianEM"]
