"""Fit and score one filtering baseline on a list of masked series.

The training and eval entry points are not called. Metrics are the shared
MSE / spectrum / KLD helpers, via ``metrics_adapter``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from dvae.baselines.filtering.data_io import window_series
from dvae.baselines.filtering.kalman_em import LinearGaussianEM
from dvae.baselines.filtering.lorenz_oracle import (
    ORACLE_NOTE,
    LorenzOracleFilter,
    integrate_lorenz,
)
from dvae.baselines.filtering.metrics_adapter import (
    aggregate_windows,
    half_half_auto_mask,
    mse_on_missing,
    score_otf_window,
)
from dvae.baselines.filtering.plotting import plot_metric_bars, plot_trajectory_window
from dvae.baselines.filtering.tresp import TrespNonlinearModel

Array = np.ndarray


def simulate_masked_lorenz(
    *,
    n_steps: int = 400,
    mask_rate: float = 0.3,
    observation_process: str = "only_x",
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
    init: Sequence[float] = (1.0, 1.0, 1.0),
    seed: int = 0,
) -> dict:
    """Short deterministic Lorenz path with a random mask. No data files."""
    from dvae.baselines.filtering.lorenz_oracle import observation_matrix

    xyz = integrate_lorenz(n_steps, sigma, rho, beta, dt, init)
    h = observation_matrix(observation_process)
    observed_vals = xyz @ h.T
    rng = np.random.default_rng(seed)
    missing = rng.random(observed_vals.shape) < mask_rate
    # Keep a few anchors so filters have something to assimilate.
    missing[:5] = False
    # ``series`` stays complete. The mask is applied inside the fit/impute
    # calls. Imputation MSE needs the held-out values.
    series = observed_vals.copy()
    return {
        "train_series": [series],
        "train_missing": [missing],
        "test_series": [series],
        "test_missing": [missing],
        "dt": dt,
        "observation_process": observation_process,
        "dataset_name": "Lorenz63",
        "lorenz_params": (sigma, rho, beta, dt),
        "xyz": xyz,
    }


def _fit_kalman(train, train_missing, cfg: Dict[str, Any]) -> LinearGaussianEM:
    obs_dim = int(np.asarray(train[0]).reshape(len(train[0]), -1).shape[1])
    state_dim = int(cfg.get("state_dim") or obs_dim)
    model = LinearGaussianEM(
        state_dim=state_dim,
        obs_dim=obs_dim,
        estimate_emission=bool(cfg.get("estimate_emission", True)),
        r_structure=str(cfg.get("r_structure", "diagonal")),
        standardize=bool(cfg.get("standardize", True)),
        n_iter=int(cfg.get("em_iters", 25)),
        tol=float(cfg.get("em_tol", 1e-4)),
        seed=int(cfg.get("seed", 0)),
    )
    init = str(cfg.get("kalman_init", "ols"))
    observed = [~np.asarray(mask, dtype=bool) for mask in train_missing]
    model.fit(train, observed, init=init)
    return model


def _fit_tresp(train, train_missing, cfg: Dict[str, Any]) -> TrespNonlinearModel:
    # Pool observed transitions across series (stays), fitting one transition.
    obs_dim = int(np.asarray(train[0]).reshape(len(train[0]), -1).shape[1])
    delay = int(cfg.get("delay", 3 if obs_dim == 1 else 1))
    model = TrespNonlinearModel(
        obs_dim=obs_dim,
        delay=delay,
        hidden=int(cfg.get("hidden", 16)),
        r_obs=float(cfg.get("r_obs", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        seed=int(cfg.get("seed", 0)),
        standardize=bool(cfg.get("standardize", True)),
    )
    # Concatenate with a break of `delay` missing steps so a window cannot
    # cross series. The missing bridge is never a fully observed training pair.
    blocks = []
    masks = []
    bridge = np.full((delay, obs_dim), np.nan, dtype=np.float64)
    bridge_m = np.ones_like(bridge, dtype=bool)
    for i, (y, m) in enumerate(zip(train, train_missing)):
        y = np.asarray(y, dtype=np.float64)
        m = np.asarray(m, dtype=bool)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            m = m.reshape(-1, 1)
        if i:
            blocks.append(bridge)
            masks.append(bridge_m)
        blocks.append(y)
        masks.append(m)
    pooled = np.concatenate(blocks, axis=0)
    pooled_m = np.concatenate(masks, axis=0)
    model.fit(
        pooled,
        ~pooled_m,
        train_mode=str(cfg.get("tresp_train", "observed_targets")),
        epochs=int(cfg.get("epochs", 40)),
        lr=float(cfg.get("lr", 1e-2)),
        n_outer=int(cfg.get("n_outer", 2)),
        gibbs_sweeps=int(cfg.get("gibbs_sweeps", 3)),
        n_sir=int(cfg.get("n_sir", 12)),
    )
    return model


def _impute_method(method, y, missing, auto_mask, cfg):
    observed = ~np.asarray(missing, dtype=bool)
    if isinstance(method, LinearGaussianEM):
        return method.impute(y, observed, auto_mask=auto_mask)
    if isinstance(method, TrespNonlinearModel):
        return method.impute(
            y,
            observed,
            auto_mask=auto_mask,
            n_particles=int(cfg.get("n_particles", 0)),
            n_samples=int(cfg.get("n_samples", 16)),
            gibbs_sweeps=int(cfg.get("gibbs_sweeps", 4)),
            gibbs_burn=int(cfg.get("gibbs_burn", 1)),
            n_sir=int(cfg.get("n_sir", 12)),
            do_ml=bool(cfg.get("do_ml", True)),
            do_gibbs=bool(cfg.get("do_gibbs", True)),
        )
    if isinstance(method, LorenzOracleFilter):
        return method.impute(y, observed, auto_mask=auto_mask)
    raise TypeError(f"Unknown method object {type(method)}")


def run_method(
    method_name: str,
    bundle: dict,
    cfg: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    """Fit on the training series and score non-overlapping test windows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train = bundle["train_series"]
    train_missing = bundle["train_missing"]
    test = bundle.get("test_series", train)
    test_missing = bundle.get("test_missing", train_missing)
    obs_name = bundle["observation_process"]
    dataset_name = bundle["dataset_name"]
    dt = float(bundle["dt"])
    seq_len = int(cfg.get("metrics_seq_len", 200))
    max_windows = int(cfg.get("max_windows", 5))
    compute_geometry = bool(cfg.get("compute_geometry", True))
    time_delay = cfg.get("time_delay")
    delay_dims = cfg.get("delay_dims")

    role = "baseline"
    note = ""
    if method_name == "kalman":
        method = _fit_kalman(train, train_missing, cfg)
        note = (
            "Linear-Gaussian SSM, Shumway & Stoffer 1982 EM, partial-observation "
            "Kalman filter and RTS smoother."
        )
    elif method_name == "tresp":
        method = _fit_tresp(train, train_missing, cfg)
        note = (
            "Tresp & Hofmann 1998: learned MLP transition, EKF filter, Gibbs "
            "smoothing, ML substitution, section 5.1 K-step sampling. "
            f"train_mode={cfg.get('tresp_train', 'observed_targets')}."
        )
    elif method_name in ("oracle_ekf", "oracle_ukf"):
        params = bundle.get("lorenz_params")
        if params is None:
            params = (
                float(cfg["sigma"]),
                float(cfg["rho"]),
                float(cfg["beta"]),
                float(cfg.get("dt", dt)),
            )
        kind = "ukf" if method_name.endswith("ukf") else "ekf"
        method = LorenzOracleFilter(
            observation_process=obs_name,
            sigma=params[0],
            rho=params[1],
            beta=params[2],
            dt=params[3],
            kind=kind,
            q_var=float(cfg.get("q_var", 1e-4)),
            r_var=float(cfg.get("r_var", 1e-3)),
        )
        role = "oracle"
        note = ORACLE_NOTE
    else:
        raise ValueError(
            f"Unknown method {method_name!r}. "
            "Expected kalman, tresp, oracle_ekf, or oracle_ukf."
        )

    windows: List[tuple] = []
    for y, missing in zip(test, test_missing):
        windows.extend(window_series(y, missing, seq_len, max_windows - len(windows)))
        if len(windows) >= max_windows:
            break
    if not windows:
        raise ValueError("No test windows. Lower metrics_seq_len or provide a longer series.")

    per_window = []
    saved_window = None
    for i, (y, missing) in enumerate(windows):
        y = np.asarray(y, dtype=np.float64)
        missing = np.asarray(missing, dtype=bool)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            missing = missing.reshape(-1, 1)
        # Ground truth is the series before NaNs were written, when the caller
        # stored NaNs in ``y``. Reconstruct GT by leaving finite entries and
        # requiring the caller to pass complete values in ``y`` with ``missing``
        # marking held-out positions. If the loader stored NaNs, GT at missing
        # positions is unknown and imputation MSE is nan — Lorenz and the demo
        # pass complete values plus a mask. Loaders that only have NaNs should
        # put the complete series in ``y`` and the mask alongside.
        gt = y.copy()
        auto_mask, flip = half_half_auto_mask(
            y.shape[0], cfg.get("flip_point")
        )
        traces = _impute_method(method, np.where(missing, np.nan, y), missing, auto_mask, cfg)
        metrics = score_otf_window(
            gt,
            traces["predicted"],
            traces["autonomous"],
            missing,
            dataset_name=dataset_name,
            observation_process=obs_name,
            dt=dt,
            flip_point=flip,
            time_delay=time_delay,
            delay_dims=delay_dims,
            compute_geometry=compute_geometry,
            filtered=traces.get("filtered"),
            smoothed=traces.get("smoothed"),
        )
        if "gibbs" in traces:
            metrics["mse_impute_gibbs"] = mse_on_missing(gt, traces["gibbs"], missing)
        if "ml" in traces:
            metrics["mse_impute_ml"] = mse_on_missing(gt, traces["ml"], missing)
        per_window.append(metrics)
        if i == 0:
            saved_window = (gt, missing, traces, flip, metrics)

    summary = aggregate_windows(per_window)
    summary["method"] = method_name
    summary["role"] = role
    summary["note"] = note
    summary["dataset_name"] = dataset_name
    summary["observation_process"] = obs_name
    summary["n_windows"] = len(per_window)
    summary["metrics_seq_len"] = int(windows[0][0].shape[0])

    # JSON cannot store numpy types.
    def _jsonify(obj):
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            val = float(obj)
            if not np.isfinite(val):
                return None
            return val
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, str):
            return obj
        return str(obj)

    (out_dir / "metrics.json").write_text(json.dumps(_jsonify(summary), indent=2))
    (out_dir / "windows.json").write_text(
        json.dumps([_jsonify(m) for m in per_window], indent=2)
    )

    if saved_window is not None and bool(cfg.get("save_plots", True)):
        gt, missing, traces, flip, metrics = saved_window
        plot_keys = [
            name
            for name in (
                "predicted",
                "filtered",
                "smoothed",
                "autonomous",
                "gibbs",
                "ml",
                "particle",
            )
            if name in traces
        ]
        plot_trajectory_window(
            gt,
            {name: traces[name] for name in plot_keys},
            missing,
            out_dir / "window0_traces.png",
            flip_point=flip,
            title=method_name,
        )
        plot_metric_bars(
            metrics,
            [
                "mse_tf",
                "mse_auto",
                "mse_impute_filtered",
                "mse_impute_smoothed",
                "mse_impute_gibbs",
                "spectrum_error_tf",
                "spectrum_error_auto",
                "kld_tf",
                "kld_auto",
            ],
            out_dir / "window0_metrics.png",
        )
        np.savez_compressed(
            out_dir / "window0_traces.npz",
            gt=gt,
            missing=missing,
            **{
                name: np.asarray(traces[name])
                for name in plot_keys
                if isinstance(traces[name], np.ndarray)
            },
        )
    if isinstance(method, LinearGaussianEM):
        np.savez_compressed(
            out_dir / "kalman_params.npz",
            F=method.F,
            H=model_H(method),
            Q=method.Q,
            R=method.R,
            mu0=method.mu0,
            P0=method.P0,
            obs_mean=method.obs_mean,
            obs_std=method.obs_std,
            loglik=np.asarray(method.loglik_history, dtype=np.float64),
        )
    return summary


def model_H(method: LinearGaussianEM) -> Array:
    return method.H


def load_bundle_from_cfg(cfg: Dict[str, Any]) -> dict:
    dataset = cfg.get("dataset", "Lorenz63")
    if dataset == "demo":
        return simulate_masked_lorenz(
            n_steps=int(cfg.get("demo_steps", 300)),
            mask_rate=float(cfg.get("mask_rate", 0.3)),
            observation_process=str(cfg.get("observation_process", "only_x")),
            sigma=float(cfg.get("sigma", 10.0)),
            rho=float(cfg.get("rho", 28.0)),
            beta=float(cfg.get("beta", 8.0 / 3.0)),
            dt=float(cfg.get("dt", 0.01)),
            seed=int(cfg.get("seed", 0)),
        )
    if dataset == "npz":
        from dvae.baselines.filtering.data_io import load_npz_series

        loaded = load_npz_series(cfg["npz"])
        loaded["train_series"] = loaded["series"]
        loaded["train_missing"] = loaded["missing"]
        loaded["test_series"] = loaded["series"]
        loaded["test_missing"] = loaded["missing"]
        return loaded
    if dataset == "Lorenz63":
        from dvae.baselines.filtering.data_io import load_lorenz63_series

        train = load_lorenz63_series(
            cfg["data_dir"],
            dataset_label=cfg.get("dataset_label"),
            mask_label=cfg.get("mask_label"),
            split="train",
            observation_process=cfg["observation_process"],
        )
        test = load_lorenz63_series(
            cfg["data_dir"],
            dataset_label=cfg.get("dataset_label"),
            mask_label=cfg.get("mask_label"),
            split="test",
            observation_process=cfg["observation_process"],
        )
        # Lorenz files store NaNs where the mask hit, and the complete xyz
        # alongside. Rebuild the observed GT from xyz so imputation MSE has
        # a target. ``series`` currently has NaNs; replace with the sliced
        # complete trajectory and keep the missing mask.
        def _gt(loaded):
            from dvae.dataset.lorenz63_dataset import Lorenz63

            holder = Lorenz63.__new__(Lorenz63)
            holder.observation_process = loaded["observation_process"]
            gt = np.asarray(
                holder.apply_observation_process(loaded["xyz"]), dtype=np.float64
            )
            if gt.ndim == 1:
                gt = gt.reshape(-1, 1)
            return gt

        train_gt = _gt(train)
        test_gt = _gt(test)
        return {
            "train_series": [train_gt],
            "train_missing": train["missing"],
            "test_series": [test_gt],
            "test_missing": test["missing"],
            "dt": train["dt"],
            "observation_process": train["observation_process"],
            "dataset_name": "Lorenz63",
            "lorenz_params": train["lorenz_params"],
        }
    if dataset == "PhysioNet2012":
        from dvae.baselines.filtering.data_io import load_physionet2012_stays

        train = load_physionet2012_stays(
            cfg["data_dir"],
            dataset_label=cfg.get("dataset_label"),
            split="train",
            observation_process=cfg.get("observation_process", "raw_vitals"),
            seq_len=int(cfg.get("metrics_seq_len", 48)),
            val_indices=float(cfg.get("val_indices", 0.2)),
        )
        test = load_physionet2012_stays(
            cfg["data_dir"],
            dataset_label=cfg.get("dataset_label"),
            split="test",
            observation_process=cfg.get("observation_process", "raw_vitals"),
            seq_len=int(cfg.get("metrics_seq_len", 48)),
            val_indices=float(cfg.get("val_indices", 0.2)),
        )
        # The dataset normalizes and keeps NaNs. There is no hidden complete
        # target beyond the recorded vitals, so GT is the series itself and
        # imputation MSE is only defined where a value was recorded and then
        # masked — which is not how this file stores data. mse_impute_* will
        # be nan unless a future export provides a dense target. OTF-style
        # mse_tf / mse_auto still score finite observed samples.
        return {
            "train_series": train["series"],
            "train_missing": train["missing"],
            "test_series": test["series"],
            "test_missing": test["missing"],
            "dt": train["dt"],
            "observation_process": train["observation_process"],
            "dataset_name": "PhysioNet2012",
            "lorenz_params": None,
        }
    raise ValueError(
        f"Unknown dataset {dataset!r}. Expected demo, npz, Lorenz63, or PhysioNet2012."
    )
