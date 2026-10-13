"""End-to-end pin: epochs whose spectrum validation is all NaN.

A tiny RNN trains a few epochs on synthetic Lorenz-shaped data with
val_criteria recon,spectrum. The spectrum ground truth is blanked to NaN
on chosen epochs, so every (window, channel) segment is skipped and the
epoch's spectrum is NaN. Those epochs must not become best, must not
reset patience, and when no epoch is ever finite the last weights are
saved with a loud warning.
"""

from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from tests.test_train_smoke import _base_config, _write_lorenz


def _train(tmp_path, monkeypatch, nan_calls, epochs=5):
    import dvae.eval.utils.spectrum_validation as sv
    from dvae.learning_algo import LearningAlgorithm

    original = sv.free_run_spectrum_error
    calls = {"n": 0}

    def blanked(predicted, ground_truth, tf_steps, *args, **kwargs):
        idx = calls["n"]
        calls["n"] += 1
        if nan_calls == "all" or idx in nan_calls:
            ground_truth = [np.full_like(np.asarray(g, dtype=float), np.nan) for g in ground_truth]
        return original(predicted, ground_truth, tf_steps, *args, **kwargs)

    monkeypatch.setattr(sv, "free_run_spectrum_error", blanked)

    data_root = tmp_path / "data"
    saved_root = tmp_path / "saved"
    _write_lorenz(data_root)
    cfg_path = tmp_path / "cfg.ini"
    cfg_path.write_text(
        _base_config(epochs=str(epochs), early_stop_patience="100", lr="0.001")
    )
    algo = LearningAlgorithm(
        {
            "cfg": str(cfg_path),
            "job_id": "nanpin",
            "reload": False,
            "device_config": {
                "saved_root": str(saved_root),
                "data_dir": str(data_root),
                "device_name": "cpu",
            },
        }
    )
    algo.build_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algo.train()
    runs = list(saved_root.rglob("loss_model.pckl"))
    assert len(runs) == 1
    run_dir = runs[0].parent
    with open(run_dir / "loss_model.pckl", "rb") as handle:
        history = pickle.load(handle)
    return run_dir, history, caught, calls["n"], algo


def test_nan_spectrum_epochs_never_best_and_patience_counts(tmp_path, monkeypatch):
    run_dir, history, caught, n_calls, algo = _train(tmp_path, monkeypatch, {0, 1}, epochs=5)
    spec = np.asarray(history["val_spectrum"], dtype=float)
    assert n_calls == spec.size == 5
    assert np.isnan(spec[:2]).all()
    assert np.isfinite(spec[2:]).all()
    skipped = np.asarray(history["val_spectrum_skipped"], dtype=float)
    assert (skipped[:2] > 0).all() and (skipped[2:] == 0).all()
    patience = np.asarray(history["cpt_patience_epochs"], dtype=float)
    # Skipped epochs increment patience; the first finite epoch is the first best.
    assert patience[0] == 1 and patience[1] == 2
    assert patience[2] == 0
    best = np.asarray(history["best_state_epochs"], dtype=float)
    assert not np.isin(best[2:], [0, 1]).any()
    finals = [p.name for p in run_dir.glob("*_final_epoch*.pt")]
    assert len(finals) == 1
    final_epoch = int(finals[0].split("_final_epoch")[1].split(".")[0])
    assert final_epoch >= 2
    # Reference set at the first finite-spectrum epoch, not epoch 0.
    assert algo._val_reference is not None
    assert np.isfinite(float(algo._val_reference["spectrum"]))
    assert not any("NO EPOCH HAD A FINITE SPECTRUM" in str(w.message) for w in caught)


def test_all_nan_spectrum_falls_back_to_last_epoch(tmp_path, monkeypatch):
    run_dir, history, caught, _, algo = _train(tmp_path, monkeypatch, "all", epochs=4)
    spec = np.asarray(history["val_spectrum"], dtype=float)
    assert np.isnan(spec).all()
    patience = np.asarray(history["cpt_patience_epochs"], dtype=float)
    assert list(patience[:4]) == [1, 2, 3, 4]
    finals = [p.name for p in run_dir.glob("*_final_epoch*.pt")]
    assert finals == [f for f in finals if f.endswith("_final_epoch3.pt")] and len(finals) == 1
    assert algo._val_reference is None
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "NO EPOCH HAD A FINITE SPECTRUM" in str(w.message)
        for w in caught
    )
