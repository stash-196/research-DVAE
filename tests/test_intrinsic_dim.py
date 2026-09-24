"""Intrinsic dimension and GT delay-dimension selection.

Synthetic arrays only. Fixed mode must keep ``_get_delay_params``.
``twonn_scan`` writes one cache on the experiment root and sibling runs
load it without scanning again.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from dvae.eval.utils.delay_dim_selection import (
    CACHE_FILENAME,
    delay_choice_summary_fields,
    load_gt_delay_cache,
    make_gt_fingerprint,
    resolve_delay_embedding,
    resolve_experiment_root,
    save_gt_delay_cache,
    scan_gt_delay_dims,
)
from dvae.eval.utils.delay_params import _get_delay_params
from dvae.eval.utils.intrinsic_dim import (
    delay_dim_for_key,
    id_metrics_for_clouds,
    participation_ratio,
    select_plateau_m,
    twonn_id,
    twonn_id_from_ratios,
)
from dvae.eval.utils.metrics_aggregate import (
    flatten_analysis_to_batch_metrics,
    merge_batch_metric_dicts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _isotropic_subspace(n, k, ambient, rng):
    basis, _ = np.linalg.qr(rng.normal(size=(ambient, k)))
    return rng.normal(size=(n, k)) @ basis.T


def test_participation_ratio_recovers_subspace_dimension():
    rng = np.random.default_rng(0)
    cloud = _isotropic_subspace(n=4000, k=3, ambient=12, rng=rng)
    pr = participation_ratio(cloud)
    assert abs(pr - 3.0) < 0.2

    with_nan = cloud.copy()
    with_nan[::7, 0] = np.nan
    pr_nan = participation_ratio(with_nan)
    assert abs(pr_nan - 3.0) < 0.25

    flat = np.ones((50, 4))
    assert np.isnan(participation_ratio(flat))


def test_twonn_from_ratios_recovers_known_dimension():
    rng = np.random.default_rng(1)

    def sample_mu(dim, n):
        # F(μ) = 1 - μ^{-d}  =>  μ = (1-U)^{-1/d}
        uniform = rng.uniform(1e-8, 1.0 - 1e-8, size=n)
        return (1.0 - uniform) ** (-1.0 / dim)

    est2 = twonn_id_from_ratios(sample_mu(2.0, 8000))
    est4 = twonn_id_from_ratios(sample_mu(4.0, 8000))
    assert abs(est2 - 2.0) < 0.25
    assert abs(est4 - 4.0) < 0.35
    assert est4 > est2


def test_twonn_id_orders_simple_point_clouds():
    rng = np.random.default_rng(2)
    id_line = twonn_id(rng.normal(size=(2500, 1)))
    id_plane = twonn_id(rng.normal(size=(2500, 2)))
    assert 0.4 < id_line < 1.7
    assert 1.2 < id_plane < 3.2
    assert id_plane > id_line + 0.3

    too_small = twonn_id(rng.normal(size=(5, 2)))
    assert np.isnan(too_small)


def test_id_metrics_use_the_same_cloud():
    rng = np.random.default_rng(3)
    cloud = _isotropic_subspace(n=3000, k=2, ambient=6, rng=rng)
    metrics = id_metrics_for_clouds(cloud, cloud, None)
    assert abs(metrics["id_pr_gt"] - 2.0) < 0.25
    assert abs(metrics["id_pr_tf"] - metrics["id_pr_gt"]) < 1e-9
    assert np.isnan(metrics["id_pr_auto"])
    assert np.isnan(metrics["id_twonn_auto"])
    assert np.isfinite(metrics["id_twonn_gt"])


def test_plateau_picker_smallest_stable_m():
    ms = [1, 2, 3, 4, 5, 6, 7]
    # Relative steps drop below 5% starting at m=3, for two consecutive steps.
    ids = [1.0, 2.0, 3.5, 3.55, 3.52, 3.50, 3.51]
    m_star, found = select_plateau_m(ms, ids, eps=0.05, k_consecutive=2)
    assert found is True
    assert m_star == 3

    m_one, found_one = select_plateau_m(ms, ids, eps=0.05, k_consecutive=1)
    assert found_one is True
    assert m_one == 3


def test_plateau_picker_fallback_when_id_keeps_growing():
    ms = [1, 2, 3, 4, 5, 6]
    ids = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    m_star, found = select_plateau_m(ms, ids, eps=0.05, k_consecutive=2)
    assert found is False
    # Smallest relative step is the last finite step, m=5 -> 6.
    assert m_star == 5


def test_plateau_picker_skips_non_finite_and_empty():
    m_star, found = select_plateau_m(
        [1, 2, 3, 4],
        [1.0, 2.0, 2.01, np.nan],
        eps=0.05,
        k_consecutive=1,
    )
    assert found is True
    assert m_star == 2

    missing, missing_found = select_plateau_m([1, 2], [np.nan, np.nan])
    assert missing is None
    assert missing_found is False


def test_fixed_mode_matches_hardcoded_delay_params(tmp_path):
    for name in ("Lorenz63", "Xhro", "anything-else"):
        tau, delay_dims = _get_delay_params(name)
        choice = resolve_delay_embedding(name, method="fixed", run_dir=str(tmp_path))
        assert (choice.time_delay, choice.delay_dims) == (tau, delay_dims)
        assert choice.source == "fixed"
        assert choice.method == "fixed"
        assert choice.delay_dims_by_channel == {}

    assert _get_delay_params("Lorenz63") == (10, 3)
    assert _get_delay_params("Xhro") == (5, 3)

    overridden = resolve_delay_embedding(
        "Lorenz63",
        method="fixed",
        time_delay_override=2,
        delay_dims_override=7,
    )
    assert (overridden.time_delay, overridden.delay_dims) == (2, 7)

    # The benchmark helper re-exports the same function (no second table).
    source = (REPO_ROOT / "src/dvae/eval/utils/benchmark_signals.py").read_text()
    assert "from dvae.eval.utils.delay_params import _get_delay_params" in source
    assert "def _get_delay_params" not in source


def test_fixed_mode_does_not_touch_cache(tmp_path):
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        raise AssertionError("fixed mode must not scan GT")

    run = tmp_path / "run_a"
    run.mkdir()
    (run / "config.ini").write_text("[DataFrame]\n")
    choice = resolve_delay_embedding(
        "Xhro",
        method="fixed",
        run_dir=str(run),
        gt_segments_loader=loader,
    )
    assert calls["n"] == 0
    assert choice.source == "fixed"
    assert not (tmp_path / CACHE_FILENAME).exists()
    assert (choice.time_delay, choice.delay_dims) == _get_delay_params("Xhro")


def _sine_segments(n=1600, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 40.0, n)
    ch1 = np.sin(t) + 0.01 * rng.normal(size=n)
    ch2 = np.cos(0.7 * t) + 0.01 * rng.normal(size=n)
    return {"ch1": [ch1], "ch2": [ch2]}


def test_scan_cache_is_shared_by_sibling_runs(tmp_path):
    experiment = tmp_path / "experiment"
    run_a = experiment / "run_a"
    run_b = experiment / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir()
    (run_a / "config.ini").write_text("[DataFrame]\nx_dim = 2\n")
    (run_b / "config.ini").write_text("[DataFrame]\nx_dim = 2\n")
    (run_a / "model_final.pt").write_bytes(b"pt")
    (run_b / "model_final.pt").write_bytes(b"pt")

    calls = {"n": 0}
    segments = _sine_segments()

    def loader():
        calls["n"] += 1
        return segments

    fingerprint = make_gt_fingerprint(
        dataset_name="Synthetic",
        seq_len=1600,
        n_windows=1,
        auto_eval_mode="half_half",
        observation_process="ch1ch2",
    )
    first = resolve_delay_embedding(
        "Synthetic",
        method="twonn_scan",
        run_dir=str(run_a),
        m_max=6,
        fingerprint=fingerprint,
        gt_segments_loader=loader,
    )
    cache_path = experiment / CACHE_FILENAME
    assert calls["n"] == 1
    assert first.source == "computed"
    assert cache_path.is_file()
    assert not any(experiment.glob("*.yaml.tmp"))
    assert not any(run_a.glob(CACHE_FILENAME))
    assert first.primary_channel == "ch1"
    assert first.delay_dims == first.delay_dims_by_channel["ch1"]
    assert first.joint_delay_dims is not None
    assert set(first.delay_dims_by_channel) == {"ch1", "ch2"}

    loaded = load_gt_delay_cache(cache_path)
    assert loaded["method"] == "twonn_scan"
    assert loaded["time_delay"] == _get_delay_params("Synthetic")[0]
    assert loaded["m_max"] == 6
    assert loaded["fingerprint"]["seq_len"] == 1600
    assert loaded["fingerprint"]["n_windows"] == 1
    assert "curves" in loaded and "ch1" in loaded["curves"]
    assert loaded["curves"]["ch1"]["m"] == [1, 2, 3, 4, 5, 6]
    # Round-trip preserves the chosen m*.
    assert loaded["delay_dims_selected"]["ch1"] == first.delay_dims_by_channel["ch1"]

    second = resolve_delay_embedding(
        "Synthetic",
        method="twonn_scan",
        run_dir=str(run_b),
        m_max=6,
        fingerprint=fingerprint,
        gt_segments_loader=loader,
    )
    assert calls["n"] == 1
    assert second.source == "cache"
    assert second.fingerprint_match is True
    assert second.delay_dims_by_channel == first.delay_dims_by_channel
    assert second.time_delay == first.time_delay
    assert second.joint_delay_dims == first.joint_delay_dims

    summary = delay_choice_summary_fields(second)
    assert summary["delay_dim_method"] == "twonn_scan"
    assert summary["delay_dims_source"] == "cache"
    assert summary["delay_dims_selected"] == second.delay_dims_by_channel
    assert summary["gt_delay_cache"] == str(cache_path)
    assert summary["gt_delay_fingerprint_match"] is True


def test_force_rescan_rewrites_cache_and_mismatch_does_not(tmp_path):
    experiment = tmp_path / "experiment"
    run = experiment / "run_a"
    run.mkdir(parents=True)
    (run / "config.ini").write_text("[DataFrame]\n")
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return _sine_segments(n=1200, seed=calls["n"])

    base_fp = make_gt_fingerprint(
        dataset_name="Synthetic",
        seq_len=1200,
        n_windows=2,
        auto_eval_mode="half_half",
    )
    resolve_delay_embedding(
        "Synthetic",
        method="twonn_scan",
        run_dir=str(run),
        m_max=4,
        fingerprint=base_fp,
        gt_segments_loader=loader,
    )
    assert calls["n"] == 1

    stale_fp = make_gt_fingerprint(
        dataset_name="Synthetic",
        seq_len=500,
        n_windows=9,
        auto_eval_mode="half_half",
    )
    cached = resolve_delay_embedding(
        "Synthetic",
        method="twonn_scan",
        run_dir=str(run),
        m_max=4,
        fingerprint=stale_fp,
        gt_segments_loader=loader,
    )
    assert calls["n"] == 1
    assert cached.source == "cache"
    assert cached.fingerprint_match is False
    assert cached.fingerprint["seq_len"] == 1200

    forced = resolve_delay_embedding(
        "Synthetic",
        method="twonn_scan",
        run_dir=str(run),
        m_max=4,
        fingerprint=stale_fp,
        gt_segments_loader=loader,
        force_rescan=True,
    )
    assert calls["n"] == 2
    assert forced.source == "computed"
    reloaded = load_gt_delay_cache(experiment / CACHE_FILENAME)
    assert reloaded["fingerprint"]["seq_len"] == 500
    assert reloaded["fingerprint"]["n_windows"] == 9


def test_cache_save_load_roundtrip_and_experiment_root(tmp_path):
    experiment = tmp_path / "sweep"
    run = experiment / "ptf_0.1"
    nested = run / "post_training_figs"
    nested.mkdir(parents=True)
    (run / "config.ini").write_text("[Network]\n")
    (experiment / "eval_logs").mkdir()
    assert resolve_experiment_root(str(run)) == str(experiment)
    assert resolve_experiment_root(str(experiment)) == str(experiment)
    assert resolve_experiment_root(str(nested)) == str(experiment)

    payload = {
        "method": "twonn_scan",
        "time_delay": 5,
        "delay_dims_selected": {"ch1": 4},
        "delay_dims": 4,
        "primary_channel": "ch1",
        "curves": {"ch1": {"m": [1, 2], "twonn": [1.0, None], "pr": [1.0, 1.2]}},
        "fingerprint": {"seq_len": 10, "n_windows": 1},
        "fingerprint_hash": "abc",
    }
    path = experiment / CACHE_FILENAME
    save_gt_delay_cache(str(path), payload)
    assert not list(experiment.glob("*.tmp"))
    loaded = yaml.safe_load(path.read_text())
    assert loaded["delay_dims_selected"]["ch1"] == 4
    assert loaded["curves"]["ch1"]["twonn"][1] is None
    assert load_gt_delay_cache(str(path))["time_delay"] == 5


def test_scan_records_pr_and_falls_back_without_finite_twonn():
    short = {"ch1": [np.linspace(0.0, 1.0, 8)]}
    scan = scan_gt_delay_dims(short, tau=5, m_max=3, fallback_delay_dims=3)
    assert scan["delay_dims_selected"]["ch1"] == 3
    assert scan["plateau_found"]["ch1"] is False
    assert scan["curves"]["ch1"]["m"] == [1, 2, 3]


def test_primary_channel_matches_batch_all_preference():
    from dvae.eval.utils.intrinsic_dim import primary_channel_key

    assert primary_channel_key(["ch1", "ch2", "ch4"]) == "ch4"
    assert primary_channel_key(["y", "x", "z"]) == "x"
    assert primary_channel_key(["ch2", "ch3"]) == "ch2"


def test_empty_gt_does_not_write_cache(tmp_path):
    run = tmp_path / "run_a"
    run.mkdir(parents=True)
    (run / "config.ini").write_text("[DataFrame]\n")

    def loader():
        return {"ch1": [np.array([], dtype=np.float64)]}

    choice = resolve_delay_embedding(
        "Xhro",
        method="twonn_scan",
        run_dir=str(run),
        gt_segments_loader=loader,
        fingerprint=make_gt_fingerprint(
            dataset_name="Xhro",
            seq_len=10,
            n_windows=1,
            auto_eval_mode="all_0",
        ),
    )
    assert choice.source == "fallback_fixed"
    assert (choice.time_delay, choice.delay_dims) == _get_delay_params("Xhro")
    assert not (tmp_path / CACHE_FILENAME).exists()


def test_delay_dim_for_key_prefers_channel_map():
    assert delay_dim_for_key(3, {"ch1": 5, "x": 2}, "ch1") == 5
    assert delay_dim_for_key(3, {"ch1": 5}, "y") == 3
    assert delay_dim_for_key(3, None, "y") == 3


def test_flatten_and_merge_keep_id_scalars():
    geom = {
        "kld_tf": 0.2,
        "kld_auto": 0.4,
        "id_pr_gt": 2.5,
        "id_pr_tf": 2.4,
        "id_pr_auto": 2.6,
        "id_twonn_gt": 1.8,
        "id_twonn_tf": 1.7,
        "id_twonn_auto": 1.9,
        "id_twonn_joint_gt": 2.2,
        "per_channel": {
            "ch1": {
                "kld_tf": 0.2,
                "kld_auto": 0.4,
                "id_pr_gt": 2.5,
                "id_twonn_auto": 1.9,
            },
            "ch2": {
                "kld_tf": 0.3,
                "kld_auto": 0.5,
                "id_pr_gt": 3.5,
                "id_twonn_auto": float("nan"),
            },
        },
    }
    flat = flatten_analysis_to_batch_metrics({}, geom, {})
    assert flat["id_pr_gt"] == 2.5
    assert flat["id_twonn_gt"] == 1.8
    assert flat["id_twonn_joint_gt"] == 2.2
    assert flat["id_pr_gt_ch1"] == 2.5
    assert flat["id_pr_gt_ch2"] == 3.5
    assert flat["id_twonn_auto_ch1"] == 1.9
    assert np.isnan(flat["id_twonn_auto_ch2"])
    assert flat["kld_tf_ch1"] == 0.2

    other = dict(flat)
    other["id_pr_gt"] = 3.5
    merged = merge_batch_metric_dicts([flat, other])
    assert merged["id_pr_gt"] == pytest.approx(3.0)
    assert "id_pr_gt_std_across_batches" in merged
    assert merged["kld_tf"] == 0.2


def test_eval_signal_cli_defaults_to_fixed_delay():
    source = (REPO_ROOT / "src/dvae/eval/eval_signal.py").read_text()
    assert '--delay-dim-method' in source
    assert 'choices=["fixed", "twonn_scan"]' in source
    assert 'default="fixed"' in source
    assert "--delay-dims" in source
    assert "--time-delay" in source
    assert "--twonn-m-max" in source
    assert "--gt-delay-cache" in source
    assert "--force-gt-delay-rescan" in source
    assert "--auto-delay-dim" in source
    assert "resolve_delay_embedding(" in source
    slurm = (REPO_ROOT / "scripts/slurm/evaluation/run_eval_multiple.sh").read_text()
    assert "DELAY_DIM_METHOD" in slurm
    assert "DELAY_DIM_FLAGS" in slurm
    style = (REPO_ROOT / "src/dvae/eval/aggregate_plot_style.py").read_text()
    assert '"id_pr_gt"' in style
    assert '"id_twonn_auto"' in style
    assert '"id_twonn_joint_gt"' in style


def test_channel_benchmarks_default_delay_matches_table():
    torch = pytest.importorskip("torch")
    from dvae.eval.utils.benchmark_signals import (
        _get_delay_params as exported,
        get_channel_benchmarks,
    )

    assert exported("Lorenz63") == _get_delay_params("Lorenz63") == (10, 3)
    batch = torch.zeros(40, 1, 1)
    benchmarks = get_channel_benchmarks(
        batch_data_long=batch,
        recon_tf=np.zeros((40, 1)),
        recon_auto_warmed=np.zeros((40, 1)),
        flip_point=20,
        dataset=object(),
        batch_idx=0,
        observation_process="",
        dataset_name="Lorenz63",
    )
    assert benchmarks["time_delay"] == 10
    assert benchmarks["delay_dims"] == 3
    assert benchmarks["delay_dims_by_channel"] is None

    overridden = get_channel_benchmarks(
        batch_data_long=batch,
        recon_tf=np.zeros((40, 1)),
        recon_auto_warmed=np.zeros((40, 1)),
        flip_point=20,
        dataset=object(),
        batch_idx=0,
        observation_process="",
        dataset_name="Lorenz63",
        time_delay=4,
        delay_dims=6,
        delay_dims_by_channel={"x": 6},
        joint_delay_dims=5,
    )
    assert overridden["time_delay"] == 4
    assert overridden["delay_dims"] == 6
    assert overridden["delay_dims_by_channel"] == {"x": 6}
    assert overridden["joint_delay_dims"] == 5
