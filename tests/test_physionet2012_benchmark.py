"""PhysioNet 2012 benchmark protocol. Synthetic tensors only; no downloads."""

from __future__ import annotations

import configparser
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    sys.modules["torch"] = MagicMock()
    sys.modules["torch.utils"] = MagicMock()
    sys.modules["torch.utils.data"] = MagicMock()
    sys.modules["torch.utils.data"].Dataset = object

from dvae.dataset.physionet2012_benchmark import (  # noqa: E402
    BENCHMARK_COLUMNS,
    BENCHMARK_OBSERVATIONS,
    HOLDOUT_FRACTION,
    MORTALITY_COLUMN,
    binary_auroc,
    build_mode_input,
    fit_logistic_proba,
    holdout_on_eval_stays,
    imputation_errors,
    interpolate_by_stay,
    is_benchmark_observation,
    parse_challenge_record_benchmark,
    plot_imputation_comparison,
    records_to_benchmark_frame,
    select_holdout,
    standardization_stats,
)
from dvae.dataset.physionet2012_dataset import (  # noqa: E402
    HOURS_PER_STAY,
    VITAL_COLUMNS,
    PhysioNet2012,
    processed_parquet_path,
)


def _frame(n_stays=12, hours=HOURS_PER_STAY, columns=None, seed=0, mortality=False):
    columns = list(columns or VITAL_COLUMNS)
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stays):
        for h in range(hours):
            row = {"stay_id": f"{1000 + s}", "hour": h}
            for col in columns:
                val = float(rng.normal())
                if rng.random() < 0.2:
                    val = np.nan
                row[col] = val
            if mortality:
                row[MORTALITY_COLUMN] = float(s % 5 == 0)
            rows.append(row)
    return pd.DataFrame(rows)


def _write(root: Path, frame: pd.DataFrame, label="hourly_v1") -> None:
    out = processed_parquet_path(root, label)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)


def _kwargs(root: Path, **overrides):
    params = dict(
        data_dir=str(root),
        dataset_label="hourly_v1",
        mask_label="None",
        split="test",
        seq_len=HOURS_PER_STAY,
        x_dim=5,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.2,
        observation_process="benchmark_vitals_otf",
        device="cpu",
        overlap=False,
        shuffle=False,
    )
    params.update(overrides)
    return params


def test_benchmark_names_do_not_alias_existing_processes():
    assert not is_benchmark_observation("raw_vitals")
    assert not is_benchmark_observation("raw_vitals_interpolate")
    assert not is_benchmark_observation("raw_hr_indicate")
    assert set(BENCHMARK_OBSERVATIONS) == {
        "benchmark_otf",
        "benchmark_interpolate",
        "benchmark_indicate",
        "benchmark_vitals_otf",
        "benchmark_vitals_interpolate",
        "benchmark_vitals_indicate",
    }
    assert len(BENCHMARK_COLUMNS) == 37
    assert set(VITAL_COLUMNS).issubset(BENCHMARK_COLUMNS)


def test_select_holdout_hits_only_observed_entries():
    observed = np.array([[True, False], [True, True], [False, False]])
    mask = select_holdout(observed, fraction=0.5, seed=3)
    assert mask.shape == observed.shape
    assert not (mask & ~observed).any()
    assert mask.sum() == 1  # floor(0.5 * 3) == 1
    again = select_holdout(observed, fraction=0.5, seed=3)
    assert np.array_equal(mask, again)
    wide = np.ones((40, 1), dtype=bool)
    assert not np.array_equal(
        select_holdout(wide, fraction=0.5, seed=1),
        select_holdout(wide, fraction=0.5, seed=2),
    )


def test_holdout_skips_train_stays():
    stay = np.array(["a", "a", "b", "b", "c", "c"])
    observed = np.ones((6, 2), dtype=bool)
    mask = holdout_on_eval_stays(
        observed,
        stay,
        {"train": ["a"], "valid": ["b"], "test": ["c"]},
        fraction=HOLDOUT_FRACTION,
        seed=2012,
    )
    assert not mask[:2].any()
    assert mask[2:].any()


def test_standardization_ignores_nontrain_rows():
    values = np.array([[0.0], [0.0], [1000.0]])
    mean, std = standardization_stats(values, np.array([True, True, False]))
    assert abs(mean[0]) < 1e-8
    assert abs(std[0] - 1.0) < 1e-8  # zero variance → 1


def test_interpolate_does_not_cross_stays():
    values = np.array([[1.0], [np.nan], [100.0], [100.0]], dtype=np.float64)
    stay = np.array(["a", "a", "b", "b"])
    filled = interpolate_by_stay(values, stay)
    assert abs(filled[1, 0] - 1.0) < 1e-5


def test_input_modes_hide_holdout_and_keep_observed():
    values = np.array([[0.0], [0.0], [10.0], [0.0]], dtype=np.float32)
    observed = np.ones((4, 1), dtype=bool)
    holdout = np.array([[False], [False], [True], [False]])
    stay = np.array(["a", "a", "a", "a"])
    otf = build_mode_input(values, observed, holdout, stay, "otf")
    assert np.isnan(otf[2, 0])
    assert otf[0, 0] == 0.0
    interp = build_mode_input(values, observed, holdout, stay, "interpolate")
    assert np.isfinite(interp).all()
    assert abs(interp[2, 0] - 0.0) < 1e-5
    indicated = build_mode_input(values, observed, holdout, stay, "indicate")
    assert indicated.shape == (4, 2)
    assert indicated[2, 0] == 0.0 and indicated[2, 1] == 0.0
    assert indicated[0, 0] == 0.0 and indicated[0, 1] == 1.0


def test_imputation_errors_and_auroc_on_synthetic_scores():
    pred = np.array([[0.0, 1.0], [3.0, 0.0]])
    target = np.array([[0.0, 0.0], [0.0, 5.0]])
    holdout = np.array([[False, True], [True, False]])
    err = imputation_errors(pred, target, holdout)
    # errors 1 and 3
    assert err["n_holdout"] == 2
    assert abs(err["mae"] - 2.0) < 1e-8
    assert abs(err["mse"] - 5.0) < 1e-8
    assert binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert binary_auroc([0, 1], [0.2, 0.2]) == 0.5
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    proba = fit_logistic_proba(x, y, x, steps=300, lr=0.2, l2=0.01)
    assert binary_auroc(y, proba) == 1.0


def test_plot_uses_aggregate_style(tmp_path):
    path = tmp_path / "imputation.png"
    plot_imputation_comparison(
        [
            {"label": "otf", "mae": 0.4, "mse": 0.3},
            {"label": "linear", "mae": 0.5, "mse": 0.4},
        ],
        path,
    )
    assert path.is_file() and path.stat().st_size > 0


def test_prepare_script_writes_a_separate_label(tmp_path):
    import importlib.util

    raw = tmp_path / "raw" / "set-a"
    raw.mkdir(parents=True)
    (raw / "1.txt").write_text(
        "Time,Parameter,Value\n00:00,RecordID,1\n00:00,HR,70\n01:00,Glucose,90\n"
    )
    root = Path(__file__).resolve().parents[1]
    script = root / "bin" / "prepare_physionet2012_benchmark.py"
    spec = importlib.util.spec_from_file_location("prep_p12", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = tmp_path / "data"
    assert mod.main(["--raw-dir", str(tmp_path / "raw"), "--data-dir", str(data)]) == 0
    written = processed_parquet_path(data, "hourly_benchmark_v1")
    assert written.is_file()
    assert not processed_parquet_path(data, "hourly_v1").exists()
    frame = pd.read_parquet(written)
    assert "Glucose" in frame.columns
    assert frame["HR"].notna().any()


def test_parse_benchmark_record_keeps_extra_series_and_mortality(tmp_path):
    raw = tmp_path / "set-a"
    raw.mkdir()
    (raw / "132539.txt").write_text(
        "Time,Parameter,Value\n"
        "00:00,RecordID,132539\n"
        "00:00,Age,60\n"
        "00:07,HR,80\n"
        "01:00,Glucose,110\n"
        "02:00,Weight,70\n"
    )
    (raw / "Outcomes-a.txt").write_text(
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death\n"
        "132539,20,5,8,10,1\n"
    )
    grid = parse_challenge_record_benchmark(raw / "132539.txt")
    assert len(grid) == HOURS_PER_STAY
    assert grid.loc[grid["hour"] == 0, "HR"].iloc[0] == 80.0
    assert grid.loc[grid["hour"] == 1, "Glucose"].iloc[0] == 110.0
    assert np.isnan(grid.loc[grid["hour"] == 0, "Glucose"].iloc[0])
    frame = records_to_benchmark_frame(tmp_path)
    assert set(BENCHMARK_COLUMNS).issubset(frame.columns)
    assert frame[MORTALITY_COLUMN].iloc[0] == 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_existing_raw_vitals_path_is_unchanged(tmp_path):
    frame = _frame()
    # Force a NaN into every row so the nan filter cannot keep a window.
    frame.loc[frame["hour"] == 0, "HR"] = np.nan
    _write(tmp_path, frame)
    raw = PhysioNet2012(**_kwargs(tmp_path, observation_process="raw_vitals", split="train"))
    assert not getattr(raw, "benchmark_active", False)
    assert raw.update_sequence_length.__func__ is PhysioNet2012.update_sequence_length
    assert np.isnan(raw.seq).any()
    with pytest.raises(ValueError):
        raw.update_sequence_length(1000, minimum_nan_ratio=0.0)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_benchmark_holdout_protocol_and_shared_mask(tmp_path):
    frame = _frame(n_stays=20, mortality=True)
    _write(tmp_path, frame)
    test_otf = PhysioNet2012(**_kwargs(tmp_path))
    test_interp = PhysioNet2012(
        **_kwargs(tmp_path, observation_process="benchmark_vitals_interpolate")
    )
    test_ind = PhysioNet2012(
        **_kwargs(tmp_path, observation_process="benchmark_vitals_indicate", x_dim=10)
    )
    train = PhysioNet2012(**_kwargs(tmp_path, split="train"))
    assert train.benchmark_holdout.sum() == 0
    assert test_otf.benchmark_holdout.sum() > 0
    n_obs = int(test_otf.benchmark_observed.sum())
    n_hold = int(test_otf.benchmark_holdout.sum())
    assert n_hold == max(1, int(np.floor(HOLDOUT_FRACTION * n_obs)))
    assert not (test_otf.benchmark_holdout & ~test_otf.benchmark_observed).any()
    assert np.array_equal(test_otf.benchmark_holdout, test_interp.benchmark_holdout)
    assert np.array_equal(test_otf.benchmark_holdout, test_ind.benchmark_holdout)
    assert np.allclose(
        test_otf.benchmark_target, test_interp.benchmark_target, equal_nan=True
    )
    hidden = test_otf.benchmark_holdout | ~test_otf.benchmark_observed
    assert np.isnan(test_otf.seq[hidden]).all()
    visible = test_otf.benchmark_observed & ~test_otf.benchmark_holdout
    assert np.allclose(test_otf.seq[visible], test_otf.benchmark_target[visible])
    assert np.isfinite(test_interp.seq).all()
    assert test_ind.seq.shape[1] == 10
    assert set(np.unique(test_ind.seq[:, 1::2])).issubset({0.0, 1.0})
    # Held-out signal is 0 and the mask bit is 0.
    assert np.all(test_ind.seq[:, 0::2][test_ind.benchmark_holdout] == 0.0)
    assert np.all(test_ind.seq[:, 1::2][test_ind.benchmark_holdout] == 0.0)
    assert test_otf.benchmark_mortality is not None
    again = PhysioNet2012(**_kwargs(tmp_path))
    assert np.array_equal(test_otf.benchmark_holdout, again.benchmark_holdout)
    # Viz-path nan filter must not abort OTF, and the window stays inside one stay.
    test_otf.update_sequence_length(1000, minimum_nan_ratio=0.0)
    assert test_otf.seq_len == HOURS_PER_STAY
    blob = pickle.dumps(test_otf)
    assert len(blob) > 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_full_column_family_requires_benchmark_parquet(tmp_path):
    _write(tmp_path, _frame())
    with pytest.raises(ValueError, match="benchmark_otf"):
        PhysioNet2012(**_kwargs(tmp_path, observation_process="benchmark_otf", x_dim=37))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_unknown_benchmark_name_is_rejected(tmp_path):
    _write(tmp_path, _frame())
    with pytest.raises(ValueError, match="Unknown benchmark"):
        PhysioNet2012(**_kwargs(tmp_path, observation_process="benchmark_nope"))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_untrained_rnn_scores_finite_holdout_error(tmp_path):
    import configparser as _cp

    from dvae.dataset.physionet2012_benchmark import (
        mortality_auroc,
        predict_dataset,
        score_predictions,
    )
    from dvae.model import build_RNN

    _write(tmp_path, _frame(n_stays=12, mortality=True))
    dataset = PhysioNet2012(**_kwargs(tmp_path, seq_len=16))
    cfg = _cp.ConfigParser()
    cfg.read_string(
        """
        [Network]
        x_dim = 5
        activation = relu
        dropout_p = 0
        dense_x =
        dense_h_x =
        dim_rnn = 4
        hidden_sh_size = 4
        num_rnn = 1
        type_rnn = RNN
        [Training]
        beta = 1
        noise_std_factor = 0.1
        noise_window_size = 8
        noise_target = both
        """
    )
    model = build_RNN(cfg, device="cpu")
    predictions, hidden = predict_dataset(model, dataset, sampling_ratio=0.0, device="cpu")
    assert predictions.shape[0] == len(dataset)
    assert predictions.shape[-1] == 5
    assert hidden is not None and hidden.shape[0] == len(dataset)
    scored = score_predictions(dataset, predictions)
    assert scored["model"]["n_holdout"] > 0
    assert np.isfinite(scored["model"]["mae"])
    assert np.isfinite(scored["reference_linear"]["mae"])
    assert np.isfinite(scored["reference_mean"]["mse"])
    train = PhysioNet2012(**_kwargs(tmp_path, split="train", seq_len=16))
    _, train_hidden = predict_dataset(model, train, sampling_ratio=0.0, device="cpu")
    probed = mortality_auroc(train, train_hidden, dataset, hidden)
    assert probed["n_train"] > 0 and probed["n_test"] > 0
    assert 0.0 <= probed["auroc"] <= 1.0


def test_config_generator_writes_three_physionet_inis(tmp_path):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    script = root / "bin" / "generate_physionet2012_benchmark_configs.py"
    spec = importlib.util.spec_from_file_location("gen_p12", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "generated"
    assert mod.main(["--subset", "full", "--output-dir", str(out)]) == 0
    inis = sorted(out.glob("*.ini"))
    assert len(inis) == 3
    processes = set()
    dims = set()
    for path in inis:
        cfg = configparser.ConfigParser()
        cfg.read(path)
        assert cfg.get("DataFrame", "dataset_name") == "PhysioNet2012"
        assert cfg.get("DataFrame", "dataset_label") == "hourly_benchmark_v1"
        assert cfg.get("Training", "sampling_method") == "ptf"
        assert cfg.getfloat("Training", "sampling_ratio") == 0.0
        assert cfg.getint("DataFrame", "sequence_len") == 48
        processes.add(cfg.get("DataFrame", "observation_process"))
        dims.add(cfg.getint("Network", "x_dim"))
        assert cfg.get("DataFrame", "dataset_name") != "Lorenz63"
    assert processes == {"benchmark_otf", "benchmark_interpolate", "benchmark_indicate"}
    assert dims == {37, 74}
    vitals = tmp_path / "vitals"
    assert mod.main(["--subset", "vitals", "--output-dir", str(vitals)]) == 0
    vital_cfg = configparser.ConfigParser()
    vital_ini = next(vitals.glob("*otf*.ini"))
    vital_cfg.read(vital_ini)
    assert vital_cfg.get("DataFrame", "dataset_label") == "hourly_v1"
    assert vital_cfg.getint("Network", "x_dim") == 5
    indicated = configparser.ConfigParser()
    indicated.read(next(vitals.glob("*indicate*.ini")))
    assert indicated.getint("Network", "x_dim") == 10


def test_eval_script_help_and_reference_only(tmp_path):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    script = root / "bin" / "eval_physionet2012_benchmark.py"
    spec = importlib.util.spec_from_file_location("eval_p12", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0

    if not HAS_TORCH:
        pytest.skip("torch required for a dataset-backed eval")
    frame = _frame(n_stays=12, mortality=True)
    _write(tmp_path / "data", frame)
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    ini = cfg_dir / "cfg.ini"
    ini.write_text(
        "\n".join(
            [
                "[Network]",
                "name = RNN",
                "x_dim = 5",
                "[Training]",
                "sampling_ratio = 0.0",
                "[DataFrame]",
                "dataset_label = hourly_v1",
                "mask_label = None",
                "sequence_len = 48",
                "sample_rate = 1",
                "skip_rate = 1",
                "val_indices = 0.2",
                "observation_process = benchmark_vitals_otf",
                "overlap = False",
            ]
        )
    )
    out = tmp_path / "out"
    code = mod.main(
        [
            "--cfg",
            str(ini),
            "--data-dir",
            str(tmp_path / "data"),
            "--split",
            "test",
            "--no-mortality",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    written = list(out.glob("*.json"))
    assert len(written) == 1
    payload = json_load(written[0])
    assert payload["model"] is None
    assert payload["reference_linear"]["n_holdout"] > 0
    assert (out / "imputation_test_benchmark_vitals_otf.png").is_file()


def json_load(path: Path):
    import json

    return json.loads(path.read_text())
