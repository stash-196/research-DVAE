#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PhysioNet Challenge 2012 imputation benchmark (opt-in).

This module does not change ``raw_hr`` / ``raw_vitals`` or their
``_interpolate`` / ``_indicate`` suffixes. Those stay on
``PhysioNet2012.apply_observation_process``. Names below are a separate
path used only when ``observation_process`` is one of
``BENCHMARK_OBSERVATIONS``.

Protocol (aligned with Du et al. 2023 / SAITS where the repo already
allows it; see the README for citations and the numbers we do not claim
to match):

* Hourly grid, 48 hours, native gaps left as NaN. No pre-imputation.
* 37 time-series variables (SAITS / PyPOTS set) or the 5-vital subset
  already written by ``bin/prepare_physionet2012.py``.
* Stay split is the existing loader split: last 20% of stays are test,
  then ``val_indices`` of the remainder are validation. Stays are never
  cut in half. SAITS uses the same fractions with a random split; this
  split is the deterministic record order of the current loader.
* Held-out mask: on validation and on test separately, hide
  ``HOLDOUT_FRACTION`` (0.10) of entries that were actually observed.
  Training stays are not artificially masked (SAITS). The hidden values
  are the imputation targets. Seed ``HOLDOUT_SEED`` so every input mode
  and the eval script share one mask. The mask is applied before
  normalization statistics are computed; statistics use training-set
  observed values only. Metrics are in that standardized space (BRITS
  and SAITS both standardize and score normalized error).
* Input modes, wired through the existing RNN forcing loop
  (``sampling_method=ptf``, ``sampling_ratio=0`` so the only autonomous
  steps are missing entries):

  - ``otf``: NaN at native gaps and held-out entries. ``BaseRNN`` default
    ``impute_strategy=auto`` writes the previous prediction into NaNs,
    and ``learning_algo`` already forces the mode selector to autonomous
    wherever the batch is NaN. Observed entries stay teacher-forced.
  - ``interpolate``: per-stay ``np.interp`` after the holdout is removed.
    Same fill as ``raw_vitals_interpolate`` (bidirectional inside the
    stay, no cross-patient fill). The tensor is finite, so the existing
    teacher-forcing path is unchanged.
  - ``indicate``: interleaved ``[x0, m0, x1, m1, ...]``. Missing and
    held-out signal values are 0 and the mask channel is 0. That is the
    layout ``_is_indicate_observation`` already trains (mask channels
    forced to teacher forcing, loss on signal channels). Because the
    tensor is finite, the existing loss also sees the zeros. That is
    current indicate behavior, not a new loss.

Published GRU-D / BRITS / SAITS figures are reference points for this
task. This repo's model is a causal RNN: ``y_t`` is emitted from ``h_t``
before ``x_t`` is consumed. Bidirectional imputers are not expected to
land on the same MAE.

Reference numbers (not produced by this repo):

* Che et al., Scientific Reports 2018, Table 1 (PhysioNet set A, 33
  variables, 5-fold mortality AUC, not an imputation MAE): GRU-D
  0.8424 ± 0.012; GRU-mean 0.8162 ± 0.014.
  https://doi.org/10.1038/s41598-018-24271-9
* Cao et al., NeurIPS 2018, Table 1–2 (set A, 4000 stays, 35
  measurements, 10% of observed values removed, error on normalized
  data, MAE (MRE)): GRU-D 0.559 (77.58%), M-RNN 0.445 (61.87%), BRITS
  0.278 (38.72%). Classification AUC: GRU-D 0.834 ± 0.002, BRITS
  0.850 ± 0.002.
  https://proceedings.neurips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html
* Du et al., Expert Systems with Applications 2023 (arXiv:2202.08516),
  Table 2 (37 features, 12000 stays, standardized, 10% of observed
  values in val/test held out; MAE / RMSE / MRE): SAITS 0.186 / 0.431
  / 26.6%, BRITS 0.256 / 0.767 / 36.5%, M-RNN 0.533 / 0.776 / 76.0%.
  Table 5 downstream RNN ROC-AUC after imputation: SAITS 84.8% ± 0.2%,
  BRITS 83.5% ± 0.1%.
  https://doi.org/10.1016/j.eswa.2023.119619

The arXiv HTML of BRITS quotes BRITS health-care MAE 0.281 (39.14%)
and GRU-D AUC 0.828 ± 0.004. The figures above are the NeurIPS
proceedings tables, not that preprint.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from dvae.dataset.physionet2012_dataset import (
    HOURS_PER_STAY,
    VITAL_COLUMNS,
    parse_hhmm_to_hour,
    partition_stay_ids,
    unique_stay_ids_in_order,
)

HOLDOUT_FRACTION = 0.10
HOLDOUT_SEED = 2012
BENCHMARK_LABEL = "hourly_benchmark_v1"
MORTALITY_COLUMN = "in_hospital_death"

# SAITS / PyPOTS PhysioNet-2012 time-series set (37). BRITS reports 35
# measurements and GRU-D reports 33; neither paper lists the dropped
# names, so this set is the SAITS one, not a claim of identity.
BENCHMARK_COLUMNS = [
    "ALP",
    "ALT",
    "AST",
    "Albumin",
    "BUN",
    "Bilirubin",
    "Cholesterol",
    "Creatinine",
    "DiasABP",
    "FiO2",
    "GCS",
    "Glucose",
    "HCO3",
    "HCT",
    "HR",
    "K",
    "Lactate",
    "MAP",
    "MechVent",
    "Mg",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "Na",
    "PaCO2",
    "PaO2",
    "Platelets",
    "RespRate",
    "SaO2",
    "SysABP",
    "Temp",
    "TroponinI",
    "TroponinT",
    "Urine",
    "WBC",
    "Weight",
    "pH",
]

_STATIC_SKIP = frozenset({"RecordID", "Age", "Gender", "Height", "ICUType"})

BENCHMARK_OBSERVATIONS = {
    "benchmark_otf": ("full", "otf"),
    "benchmark_interpolate": ("full", "interpolate"),
    "benchmark_indicate": ("full", "indicate"),
    "benchmark_vitals_otf": ("vitals", "otf"),
    "benchmark_vitals_interpolate": ("vitals", "interpolate"),
    "benchmark_vitals_indicate": ("vitals", "indicate"),
}


def is_benchmark_observation(observation_process) -> bool:
    return observation_process in BENCHMARK_OBSERVATIONS


def columns_for_family(family: str) -> list[str]:
    if family == "vitals":
        return list(VITAL_COLUMNS)
    if family == "full":
        return list(BENCHMARK_COLUMNS)
    raise ValueError(f"Unknown benchmark column family {family!r}")


def select_holdout(
    observed: np.ndarray,
    fraction: float = HOLDOUT_FRACTION,
    seed: int = HOLDOUT_SEED,
) -> np.ndarray:
    """Boolean mask, True on a deterministic subset of observed entries.

    ``n_hold = floor(fraction * n_observed)``. If that is 0 but some
    entries are observed and ``fraction > 0``, one entry is held out so
    a tiny synthetic series still has an imputation target.
    """
    observed = np.asarray(observed, dtype=bool)
    mask = np.zeros(observed.shape, dtype=bool)
    candidates = np.flatnonzero(observed.ravel())
    n = int(candidates.size)
    if n == 0 or fraction <= 0:
        return mask
    n_hold = int(np.floor(float(fraction) * n))
    if n_hold <= 0:
        n_hold = 1
    n_hold = min(n_hold, n)
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(candidates, size=n_hold, replace=False)
    flat = mask.ravel()
    flat[chosen] = True
    return mask


def holdout_on_eval_stays(
    observed: np.ndarray,
    stay_ids: np.ndarray,
    split_ids_by_name: dict,
    fraction: float = HOLDOUT_FRACTION,
    seed: int = HOLDOUT_SEED,
) -> np.ndarray:
    """10% of observed entries inside validation, and 10% inside test.

    Training rows stay False. Validation uses ``seed``; test uses
    ``seed + 1`` so the two draws are independent and reproducible.
    """
    observed = np.asarray(observed, dtype=bool)
    stay_ids = np.asarray(stay_ids)
    mask = np.zeros(observed.shape, dtype=bool)
    for offset, name in ((0, "valid"), (1, "test")):
        ids = split_ids_by_name.get(name) or []
        if len(ids) == 0:
            continue
        eligible = np.isin(stay_ids, list(ids))
        if not eligible.any():
            continue
        part = select_holdout(observed & eligible[:, None], fraction, seed + offset)
        mask |= part
    return mask


def standardization_stats(values: np.ndarray, train_rows: np.ndarray):
    """Per-column mean and std on finite training entries. Empty columns → 0, 1."""
    block = np.asarray(values, dtype=np.float64)[np.asarray(train_rows)]
    mean = np.nanmean(block, axis=0)
    std = np.nanstd(block, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std >= 1e-8), std, 1.0)
    return mean.astype(np.float64), std.astype(np.float64)


def apply_standardization(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    out = (np.asarray(values, dtype=np.float64) - mean) / std
    return out.astype(np.float32)


def interpolate_by_stay(values: np.ndarray, stay_ids: np.ndarray) -> np.ndarray:
    """Linear fill inside each stay. All-NaN columns in a stay become 0.

    ``np.interp`` uses both directions inside the stay and does not read
    another stay. Same rule as ``raw_vitals_interpolate``.
    """
    values = np.asarray(values, dtype=np.float64)
    stay_ids = np.asarray(stay_ids)
    out = values.copy()
    n = len(stay_ids)
    i = 0
    while i < n:
        j = i + 1
        while j < n and stay_ids[j] == stay_ids[i]:
            j += 1
        block = out[i:j]
        idx = np.arange(block.shape[0])
        for d in range(block.shape[1]):
            col = block[:, d]
            valid = np.isfinite(col)
            if not valid.any():
                block[:, d] = 0.0
            elif (~valid).any():
                col = col.copy()
                col[~valid] = np.interp(idx[~valid], idx[valid], col[valid])
                block[:, d] = col
        out[i:j] = block
        i = j
    return out.astype(np.float32)


def build_mode_input(
    standardized: np.ndarray,
    observed: np.ndarray,
    holdout: np.ndarray,
    stay_ids: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Model input for one mode. ``standardized`` keeps NaN where unobserved."""
    standardized = np.asarray(standardized, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    holdout = np.asarray(holdout, dtype=bool)
    hidden = (~observed) | holdout
    if mode == "otf":
        out = standardized.copy()
        out[hidden] = np.nan
        return out
    if mode == "interpolate":
        source = standardized.copy()
        source[hidden] = np.nan
        return interpolate_by_stay(source, stay_ids)
    if mode == "indicate":
        signal = np.nan_to_num(standardized, nan=0.0).astype(np.float32)
        signal = signal.copy()
        signal[holdout] = 0.0
        signal[~observed] = 0.0
        present = (observed & ~holdout).astype(np.float32)
        t, c = signal.shape
        packed = np.empty((t, 2 * c), dtype=np.float32)
        packed[:, 0::2] = signal
        packed[:, 1::2] = present
        return packed
    raise ValueError(f"Unknown benchmark mode {mode!r}")


def imputation_errors(prediction, target, holdout) -> dict:
    """MAE / MSE / RMSE / MRE on held-out entries only.

    ``prediction`` and ``target`` share the signal shape. ``holdout`` is
    boolean. MRE is ``sum |err| / sum |target|`` (BRITS).
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    holdout = np.asarray(holdout, dtype=bool)
    if prediction.shape != target.shape or holdout.shape != target.shape:
        raise ValueError(
            f"Shape mismatch pred={prediction.shape} target={target.shape} "
            f"holdout={holdout.shape}"
        )
    sel = holdout & np.isfinite(target)
    pred = prediction[sel]
    gt = target[sel]
    finite = np.isfinite(pred)
    n_nonfinite = int((~finite).sum())
    pred = pred[finite]
    gt = gt[finite]
    n = int(pred.size)
    if n == 0:
        return {
            "n_holdout": 0,
            "n_nonfinite": n_nonfinite,
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mre": float("nan"),
        }
    err = pred - gt
    abs_err = np.abs(err)
    denom = float(np.abs(gt).sum())
    mse = float(np.mean(err ** 2))
    return {
        "n_holdout": n,
        "n_nonfinite": n_nonfinite,
        "mae": float(np.mean(abs_err)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mre": float(abs_err.sum() / denom) if denom > 0 else float("nan"),
    }


def binary_auroc(labels, scores) -> float:
    """Mann–Whitney AUROC. NaN if either class is absent."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must share a shape")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # 1-based average rank for the tie group
        avg = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg
        i = j
    sum_pos = float(ranks[labels == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fit_logistic_proba(x_train, y_train, x_score, l2: float = 1.0, steps: int = 400, lr: float = 0.05):
    """L2 logistic regression. Returns probabilities for ``x_score``.

    Features are standardized with training mean and std. This is a linear
    probe on frozen states, not a GRU-D / BRITS classification head.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    x_score = np.asarray(x_score, dtype=np.float64)
    if x_train.ndim != 2 or x_score.ndim != 2:
        raise ValueError("probe inputs must be 2D")
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)

    def _norm(z):
        return (z - mu) / sd

    xt = np.concatenate([_norm(x_train), np.ones((len(x_train), 1))], axis=1)
    xs = np.concatenate([_norm(x_score), np.ones((len(x_score), 1))], axis=1)
    w = np.zeros(xt.shape[1], dtype=np.float64)
    n = max(len(y_train), 1)
    for _ in range(int(steps)):
        logits = np.clip(xt @ w, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        grad = xt.T @ (prob - y_train) / n
        grad[:-1] += float(l2) * w[:-1]
        w -= float(lr) * grad
    logits = np.clip(xs @ w, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def window_slices(dataset):
    """(start, end) for every index. Windows are full ``seq_len``."""
    slices = []
    for index in range(len(dataset)):
        start = int(dataset.data_idx[index])
        end = start + int(dataset.seq_len)
        slices.append((start, end))
    return slices


def stack_windows(array, slices) -> np.ndarray:
    return np.stack([np.asarray(array[s:e]) for s, e in slices], axis=0)


def reference_linear_windows(dataset) -> np.ndarray:
    """Per-stay linear interpolation of the standardized series with holdouts removed."""
    source = np.array(dataset.benchmark_target, dtype=np.float32, copy=True)
    source[dataset.benchmark_holdout] = np.nan
    filled = interpolate_by_stay(source, dataset.stay_id_per_row)
    return stack_windows(filled, window_slices(dataset))


def reference_mean_windows(dataset) -> np.ndarray:
    """Training mean in standardized space is 0, so the mean fill is 0."""
    slices = window_slices(dataset)
    target = stack_windows(dataset.benchmark_target, slices)
    return np.zeros_like(target)


def one_window_index_per_stay(dataset) -> dict:
    """Stay id → dataset index of the window that ends latest."""
    best = {}
    for index, (start, end) in enumerate(window_slices(dataset)):
        sid = dataset.stay_id_per_row[start]
        if sid not in best or end > best[sid][0]:
            best[sid] = (end, index)
    return {sid: index for sid, (_, index) in best.items()}


def stay_labels_for_indices(dataset, indices: list[int]) -> np.ndarray:
    labels = []
    mortality = dataset.benchmark_mortality
    for index in indices:
        start = int(dataset.data_idx[index])
        if mortality is None:
            labels.append(np.nan)
        else:
            labels.append(float(mortality[start]))
    return np.asarray(labels, dtype=np.float64)


def benchmark_update_sequence_length(self, new_seq_len=None, minimum_nan_ratio=None):
    """Class ``update_sequence_length``, with a fallback for the training viz path.

    ``learning_algo`` calls ``update_sequence_length(1000, minimum_nan_ratio=0.3)``
    when it saves figures. ICU series are far more than 30% missing, so that
    filter matches no OTF window and would abort the run. Only benchmark
    instances bind this wrapper. ``raw_vitals`` keeps the class method.
    """
    from dvae.dataset.physionet2012_dataset import PhysioNet2012

    try:
        return PhysioNet2012.update_sequence_length(
            self, new_seq_len, minimum_nan_ratio
        )
    except ValueError:
        if minimum_nan_ratio is None:
            raise
        return PhysioNet2012.update_sequence_length(self, new_seq_len, None)


def init_benchmark_dataset(dataset, frame: pd.DataFrame) -> None:
    """Fill a ``PhysioNet2012`` instance and return. Existing names never call this."""
    import types

    family, mode = BENCHMARK_OBSERVATIONS[dataset.observation_process]
    columns = columns_for_family(family)
    missing_cols = [c for c in columns if c not in frame.columns]
    if missing_cols:
        raise ValueError(
            f"{dataset.observation_process} needs columns {missing_cols}. "
            "Run bin/prepare_physionet2012_benchmark.py for the 37-variable "
            f"parquet ({BENCHMARK_LABEL}), or use a benchmark_vitals_* "
            "process on the 5-vital hourly_v1 file."
        )

    fraction = float(getattr(dataset, "holdout_fraction", HOLDOUT_FRACTION))
    seed = int(getattr(dataset, "holdout_seed", HOLDOUT_SEED))
    # kwargs on the dataset constructor land in dataset.__dict__ only if
    # we copy them. PhysioNet2012 stores **kwargs unused. Read them here.
    extra = getattr(dataset, "_benchmark_kwargs", None)
    if extra:
        if extra.get("holdout_fraction") is not None:
            fraction = float(extra["holdout_fraction"])
        if extra.get("holdout_seed") is not None:
            seed = int(extra["holdout_seed"])

    values = frame[columns].to_numpy(dtype=np.float64)
    stay_ids = frame["stay_id"].to_numpy()
    observed = np.isfinite(values)
    unique_ids = unique_stay_ids_in_order(stay_ids.tolist())
    train_ids, val_ids, test_ids = partition_stay_ids(
        unique_ids, val_frac=float(dataset.val_indices)
    )
    holdout = holdout_on_eval_stays(
        observed,
        stay_ids,
        {"train": train_ids, "valid": val_ids, "test": test_ids},
        fraction=fraction,
        seed=seed,
    )
    train_rows = np.isin(stay_ids, list(train_ids))
    mean, std = standardization_stats(values, train_rows)
    standardized = apply_standardization(values, mean, std)
    # Standardization keeps NaNs as NaN.
    standardized = np.where(observed, standardized, np.nan).astype(np.float32)

    if dataset.split == "test":
        keep = test_ids
    elif dataset.split == "valid":
        keep = val_ids
    else:
        keep = train_ids
    if not keep:
        raise ValueError(
            f"No stays left for split={dataset.split!r} "
            f"(n_stays={len(unique_ids)}, val_indices={dataset.val_indices})."
        )
    keep_rows = np.isin(stay_ids, list(keep))
    frame_split = frame.loc[keep_rows].reset_index(drop=True)
    standardized = standardized[keep_rows]
    observed = observed[keep_rows]
    holdout = holdout[keep_rows]
    stay_ids = stay_ids[keep_rows]

    model_input = build_mode_input(
        standardized, observed, holdout, stay_ids, mode
    )
    n_signal = len(columns)
    expected = n_signal * 2 if mode == "indicate" else n_signal
    if dataset.x_dim is None:
        dataset.x_dim = expected
    elif int(dataset.x_dim) != expected:
        raise ValueError(
            f"Expected x_dim={expected} for {dataset.observation_process}, "
            f"got {dataset.x_dim}."
        )

    mortality = None
    if MORTALITY_COLUMN in frame_split.columns:
        mortality = (
            pd.to_numeric(frame_split[MORTALITY_COLUMN], errors="coerce")
            .to_numpy(dtype=np.float64)
        )

    dataset.full_sequence = frame_split
    dataset.stay_id_per_row = stay_ids
    dataset.missing_mask = (~observed) | holdout
    dataset.seq = model_input
    dataset.benchmark_active = True
    dataset.benchmark_mode = mode
    dataset.benchmark_family = family
    dataset.benchmark_columns = columns
    dataset.benchmark_target = standardized
    dataset.benchmark_observed = observed
    dataset.benchmark_holdout = holdout
    dataset.benchmark_mortality = mortality
    dataset.benchmark_norm_mean = mean
    dataset.benchmark_norm_std = std
    dataset.benchmark_holdout_fraction = fraction
    dataset.benchmark_holdout_seed = seed
    dataset.update_sequence_length = types.MethodType(
        benchmark_update_sequence_length, dataset
    )
    dataset.update_sequence_length(dataset.seq_len)
    n_hold = int(holdout.sum())
    print(
        f"[PhysioNet2012][{dataset.observation_process}] split={dataset.split} "
        f"rows={len(stay_ids)} holdout={n_hold} x_dim={dataset.x_dim} "
        f"mode={mode}"
    )


def predict_dataset(model, dataset, sampling_ratio: float = 0.0, batch_size: int = 32, device: str = "cpu"):
    """Run ``model`` the way training does and return signal predictions.

    Returns
    -------
    predictions : ndarray, shape (n_windows, seq_len, n_signal)
        Indicate outputs are the even channels only.
    last_hidden : ndarray or None, shape (n_windows, hidden)
        Last recurrent state of each window, when the module stores ``h``.
    """
    import torch

    from dvae.utils.model_mode_selector import create_autonomous_mode_selector

    if not getattr(dataset, "benchmark_active", False):
        raise ValueError("predict_dataset expects a benchmark PhysioNet2012 split")
    indicate = dataset.benchmark_mode == "indicate"
    model.eval()
    preds = []
    hiddens = []
    n = len(dataset)
    if n == 0:
        return np.zeros((0, dataset.seq_len, len(dataset.benchmark_columns))), None
    for start in range(0, n, int(batch_size)):
        idx = list(range(start, min(start + int(batch_size), n)))
        batch = torch.stack([dataset[i] for i in idx], dim=0).to(device)
        x = batch.permute(1, 0, 2)
        seq_len, bs, x_dim = x.shape
        if float(sampling_ratio) <= 0.0:
            base = torch.zeros(seq_len, bs, x_dim, device=x.device)
        else:
            base = create_autonomous_mode_selector(
                seq_len,
                mode="bernoulli_sampling",
                autonomous_ratio=float(sampling_ratio),
                batch_size=bs,
                x_dim=x_dim,
                device=x.device,
            )
            base = torch.as_tensor(base, device=x.device).float()
        mode = torch.where(torch.isnan(x), torch.ones_like(x), base)
        if indicate:
            mode = mode.clone()
            mode[:, :, 1::2] = 0.0
        with torch.no_grad():
            y = model(x, mode_selector=mode, noise_selector=None)
        y_np = y.detach().cpu().numpy()
        if indicate:
            y_np = y_np[:, :, 0::2]
        for b in range(y_np.shape[1]):
            preds.append(y_np[:, b, :])
        hidden = getattr(model, "h", None)
        if hidden is not None:
            h_np = hidden.detach().cpu().numpy()
            for b in range(h_np.shape[1]):
                hiddens.append(h_np[-1, b])
    predictions = np.stack(preds, axis=0)
    last_hidden = np.stack(hiddens, axis=0) if hiddens else None
    return predictions, last_hidden


def score_predictions(dataset, predictions) -> dict:
    """Imputation errors of ``predictions`` plus mean and linear references."""
    slices = window_slices(dataset)
    target = stack_windows(dataset.benchmark_target, slices)
    holdout = stack_windows(dataset.benchmark_holdout, slices)
    model_err = imputation_errors(predictions, target, holdout)
    linear_err = imputation_errors(reference_linear_windows(dataset), target, holdout)
    mean_err = imputation_errors(reference_mean_windows(dataset), target, holdout)
    return {
        "model": model_err,
        "reference_linear": linear_err,
        "reference_mean": mean_err,
    }


def mortality_auroc(train_dataset, train_hidden, test_dataset, test_hidden) -> dict:
    """Linear probe on the last hidden state of the latest window of each stay."""
    if train_hidden is None or test_hidden is None:
        return {"auroc": float("nan"), "n_train": 0, "n_test": 0, "reason": "no hidden state"}
    if train_dataset.benchmark_mortality is None or test_dataset.benchmark_mortality is None:
        return {
            "auroc": float("nan"),
            "n_train": 0,
            "n_test": 0,
            "reason": "no in_hospital_death column",
        }

    def _collect(dataset, hidden):
        chosen = one_window_index_per_stay(dataset)
        indices = list(chosen.values())
        labels = stay_labels_for_indices(dataset, indices)
        states = hidden[np.asarray(indices, dtype=int)]
        keep = np.isfinite(labels) & np.isfinite(states).all(axis=1)
        return states[keep], labels[keep].astype(int)

    x_train, y_train = _collect(train_dataset, train_hidden)
    x_test, y_test = _collect(test_dataset, test_hidden)
    if len(y_train) == 0 or len(np.unique(y_train)) < 2 or len(y_test) == 0:
        return {
            "auroc": float("nan"),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "reason": "need both classes in train and at least one labeled test stay",
        }
    scores = fit_logistic_proba(x_train, y_train, x_test)
    return {
        "auroc": float(binary_auroc(y_test, scores)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "reason": "linear probe on last hidden state",
    }


def plot_imputation_comparison(rows: list[dict], path) -> None:
    """Bar chart of MAE and MSE. Uses ``aggregate_plot_style``."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    from dvae.eval.aggregate_plot_style import apply_paper_ready_line_style, save_figure

    apply_paper_ready_line_style()
    labels = [str(row["label"]) for row in rows]
    mae = [float(row["mae"]) for row in rows]
    mse = [float(row["mse"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(labels) + 2.0), 5.0))
    ax.bar(x - width / 2, mae, width, label="MAE")
    ax.bar(x + width / 2, mse, width, label="MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Imputation error")
    ax.legend()
    save_figure(fig, str(path))
    plt.close(fig)


def collect_benchmark_records(raw_dir: Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    files = []
    for sub in ("set-a", "set-b", "set-c", "set-A", "set-B", "set-C"):
        d = raw_dir / sub
        if d.is_dir():
            files.extend(sorted(d.glob("*.txt")))
    files.extend(sorted(raw_dir.glob("*.txt")))
    seen = set()
    unique = []
    for path in files:
        if path.name.lower().startswith("outcomes"):
            continue
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def parse_challenge_record_benchmark(path: Path) -> pd.DataFrame:
    """One Challenge CSV → 48 hourly rows. All 37 series; gaps stay NaN."""
    rows = []
    stay_id = path.stem
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty record: {path}")
        header_l = [h.strip() for h in header]
        if header_l[:3] == ["Time", "Parameter", "Value"]:
            time_i, param_i, value_i = 0, 1, 2
        elif "Time" in header_l and "Parameter" in header_l and "Value" in header_l:
            time_i = header_l.index("Time")
            param_i = header_l.index("Parameter")
            value_i = header_l.index("Value")
        else:
            raise ValueError(f"Unrecognized header {header_l} in {path}")
        allowed = set(BENCHMARK_COLUMNS)
        for parts in reader:
            if len(parts) <= max(time_i, param_i, value_i):
                continue
            param = parts[param_i].strip()
            if param == "RecordID":
                stay_id = parts[value_i].strip()
                continue
            if param in _STATIC_SKIP or param not in allowed:
                continue
            hour = parse_hhmm_to_hour(parts[time_i])
            if hour is None:
                continue
            try:
                value = float(parts[value_i])
            except ValueError:
                continue
            rows.append((hour, param, value))

    grid = pd.DataFrame(
        {"stay_id": stay_id, "hour": np.arange(HOURS_PER_STAY, dtype=np.int32)}
    )
    for col in BENCHMARK_COLUMNS:
        grid[col] = np.nan
    if rows:
        tmp = pd.DataFrame(rows, columns=["hour", "param", "value"])
        last = tmp.groupby(["hour", "param"], sort=False).last().reset_index()
        pivoted = last.pivot(index="hour", columns="param", values="value")
        for col in BENCHMARK_COLUMNS:
            if col in pivoted.columns:
                grid[col] = pivoted[col].reindex(grid["hour"]).to_numpy(dtype=np.float64)
    return grid


def load_mortality_labels(raw_dir: Path) -> dict[str, float]:
    """RecordID → in-hospital death from any ``Outcomes*.txt`` under ``raw_dir``."""
    labels: dict[str, float] = {}
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.rglob("Outcomes*.txt"))
    for path in files:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                continue
            header_l = [h.strip() for h in header]
            if "RecordID" not in header_l or "In-hospital_death" not in header_l:
                continue
            id_i = header_l.index("RecordID")
            death_i = header_l.index("In-hospital_death")
            for parts in reader:
                if len(parts) <= max(id_i, death_i):
                    continue
                try:
                    labels[parts[id_i].strip()] = float(parts[death_i])
                except ValueError:
                    continue
    return labels


def records_to_benchmark_frame(raw_dir: Path) -> pd.DataFrame:
    files = collect_benchmark_records(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No Challenge 2012 .txt records under {raw_dir}. "
            "Download set-a/set-b from "
            "https://physionet.org/content/challenge-2012/1.0.0/ "
            "and extract them here."
        )
    frame = pd.concat(
        [parse_challenge_record_benchmark(path) for path in files], ignore_index=True
    )
    labels = load_mortality_labels(raw_dir)
    frame[MORTALITY_COLUMN] = frame["stay_id"].astype(str).map(labels).astype(float)
    return frame
