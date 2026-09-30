"""Joint intrinsic dimension of hidden-state trajectories.

Synthetic arrays only. Observation-space ``id_*_joint_*`` stays a separate
family. There is no ground-truth latent, so no ``*_hidden_joint_gt`` key.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dvae.eval.utils.intrinsic_dim import (
    ID_HIDDEN_JOINT_GT_NOTE,
    hidden_joint_id_from_benchmarks,
    hidden_state_cloud,
    id_metrics_for_hidden_joint,
    participation_ratio,
    snapshot_hidden_state,
)
from dvae.eval.utils.metrics_aggregate import flatten_analysis_to_batch_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]


def _subspace(n, k, ambient, rng):
    basis, _ = np.linalg.qr(rng.normal(size=(ambient, k)))
    return rng.normal(size=(n, k)) @ basis.T


class _Tensor:
    def __init__(self, arr):
        self.arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _Model:
    def __init__(self, h):
        self.h = h


def test_snapshot_hidden_state_copies_batch_trajectory():
    h = np.arange(24, dtype=np.float64).reshape(4, 2, 3)
    copied = snapshot_hidden_state(_Model(_Tensor(h)))
    assert copied.shape == (4, 2, 3)
    assert copied[0, 0, 0] == 0.0
    copied[0, 0, 0] = 99.0
    assert h[0, 0, 0] == 0.0
    assert snapshot_hidden_state(_Model(None)) is None
    assert snapshot_hidden_state(object()) is None


def test_hidden_cloud_uses_batch0_and_auto_mask():
    rng = np.random.default_rng(0)
    plane = _subspace(n=2000, k=2, ambient=8, rng=rng)
    noise = rng.normal(size=(2000, 8))
    h = np.stack([plane, noise], axis=1)
    mask = np.zeros(2000, dtype=bool)
    mask[200:] = True
    cloud = hidden_state_cloud(h, mask)
    assert cloud.shape == (1800, 8)
    # Batch 1 is full-rank noise; batch 0 is a plane.
    assert abs(participation_ratio(cloud) - 2.0) < 0.35
    assert hidden_state_cloud(h, np.ones(3, dtype=bool)) is None
    assert hidden_state_cloud(None, mask) is None


def test_hidden_joint_id_orders_dimension_and_omits_gt():
    rng = np.random.default_rng(1)
    line = _subspace(n=2500, k=1, ambient=6, rng=rng)[:, None, :]
    volume = _subspace(n=2500, k=3, ambient=6, rng=rng)[:, None, :]
    mask = np.ones(2500, dtype=bool)
    low = hidden_joint_id_from_benchmarks(
        {"hidden_tf": line, "hidden_auto": line, "auto_mask": mask}
    )
    high = hidden_joint_id_from_benchmarks(
        {"hidden_tf": volume, "hidden_auto": volume, "auto_mask": mask}
    )
    assert abs(low["id_pr_hidden_joint_tf"] - 1.0) < 0.35
    assert abs(high["id_pr_hidden_joint_auto"] - 3.0) < 0.45
    assert low["id_pr_hidden_joint_tf"] < high["id_pr_hidden_joint_tf"]
    assert np.isfinite(low["id_twonn_hidden_joint_tf"])
    assert np.isfinite(high["id_twonn_hidden_joint_auto"])
    assert "id_pr_hidden_joint_gt" not in low
    assert "id_twonn_hidden_joint_gt" not in low
    assert "id_pr_joint_tf" not in low
    assert low["id_hidden_joint_gt_note"] == ID_HIDDEN_JOINT_GT_NOTE
    assert "no ground-truth latent" in ID_HIDDEN_JOINT_GT_NOTE


def test_short_cloud_keeps_pr_and_skips_twonn():
    rng = np.random.default_rng(2)
    small = rng.normal(size=(12, 1, 5))
    metrics = id_metrics_for_hidden_joint(
        hidden_state_cloud(small),
        hidden_state_cloud(small),
    )
    assert np.isfinite(metrics["id_pr_hidden_joint_tf"])
    assert np.isnan(metrics["id_twonn_hidden_joint_auto"])
    assert set(metrics) == {
        "id_pr_hidden_joint_tf",
        "id_pr_hidden_joint_auto",
        "id_twonn_hidden_joint_tf",
        "id_twonn_hidden_joint_auto",
    }


def test_missing_trajectories_add_no_keys():
    assert hidden_joint_id_from_benchmarks({}) == {}
    assert hidden_joint_id_from_benchmarks(None) == {}
    assert hidden_joint_id_from_benchmarks({"auto_mask": np.array([True, True])}) == {}


def test_flatten_keeps_hidden_joint_scalars_and_observation_joint():
    flat = flatten_analysis_to_batch_metrics(
        {},
        {
            "id_pr_joint_gt": 1.5,
            "id_pr_hidden_joint_tf": 2.5,
            "id_twonn_hidden_joint_auto": 3.25,
            "id_hidden_joint_gt_note": ID_HIDDEN_JOINT_GT_NOTE,
        },
        {},
    )
    assert flat["id_pr_joint_gt"] == 1.5
    assert flat["id_pr_hidden_joint_tf"] == 2.5
    assert flat["id_twonn_hidden_joint_auto"] == 3.25
    assert "id_hidden_joint_gt_note" not in flat
    assert "id_pr_hidden_joint_gt" not in flat


def test_eval_wires_hidden_snapshots_without_dropping_observation_joint():
    eval_src = (REPO_ROOT / "src/dvae/eval/eval_signal.py").read_text()
    assert "snapshot_hidden_state(dvae)" in eval_src
    assert "hidden_tf=hidden_tf" in eval_src
    assert "hidden_auto=hidden_auto" in eval_src
    assert "ID_HIDDEN_JOINT_GT_NOTE" in eval_src
    geom_src = (REPO_ROOT / "src/dvae/eval/utils/run_geometry_analysis.py").read_text()
    assert "hidden_joint_id_from_benchmarks" in geom_src
    assert '"id_pr_joint_gt"' in geom_src
    style = (REPO_ROOT / "src/dvae/eval/aggregate_plot_style.py").read_text()
    assert '"id_pr_hidden_joint_tf"' in style
    assert '"id_twonn_hidden_joint_auto"' in style
