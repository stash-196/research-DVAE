"""Snapshots of training state that must not alias live parameters.

``module.state_dict()`` tensors share storage with the parameters, and
optimizer state tensors do the same. A later ``optimizer.step`` rewrites
those tensors in place, so a "best" checkpoint saved at the end of
training would otherwise hold the stopping-epoch weights. Clones break
that alias. Latest-checkpoint resume also stores the RNG here.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch

LATEST_FORMAT_VERSION = 1

# Bump-table types that advance ``auto_kl_done`` when they are a real
# autonomous or KL step. A window bump can emit the same type strings as
# a reset; those rows do not advance the counter.
_AUTO_KL_TYPES = ("auto", "KL")


def clone_module_state_dict(state_dict) -> dict:
    """Detach-and-clone tensors so later steps cannot rewrite the snapshot."""
    cloned = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def clone_optimizer_state_dict(state_dict) -> dict:
    """Deep copy of an optimizer ``state_dict``, including moment buffers."""
    return copy.deepcopy(state_dict)


def capture_rng_state() -> dict:
    """Torch, CUDA (if available), NumPy, and ``random`` states."""
    state = {
        "torch": torch.get_rng_state().clone(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [tensor.clone() for tensor in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state) -> None:
    """Put the process RNGs back. Missing CUDA state is left alone."""
    if not isinstance(state, dict):
        raise ValueError("RNG state must be a dict from capture_rng_state.")
    torch_state = state["torch"]
    if torch.is_tensor(torch_state):
        torch_state = torch_state.cpu()
    torch.set_rng_state(torch_state)
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    cuda_state = state.get("torch_cuda")
    if cuda_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _series_at(loss_log, key, index, default):
    if not isinstance(loss_log, dict) or key not in loss_log:
        return default
    values = np.asarray(loss_log[key], dtype=np.float64).reshape(-1)
    if index < 0 or index >= values.size or not np.isfinite(values[index]):
        return default
    return float(values[index])


def auto_kl_done_through_epoch(bump_events, best_epoch) -> int:
    """How many autonomous/KL bumps have already fired by ``best_epoch``.

    One increment per auto row and per KL row on epochs ``<= best_epoch``.
    Rows that share an epoch with a window bump are resets, not steps, and
    do not count. A noise-only autonomous bump (no auto row) counts once.
    """
    if not bump_events or best_epoch is None:
        return 0
    by_epoch = {}
    for row in bump_events:
        if not isinstance(row, dict) or "epoch" not in row:
            continue
        epoch = int(row["epoch"])
        if epoch > int(best_epoch):
            continue
        by_epoch.setdefault(epoch, set()).add(str(row.get("type", "")))
    done = 0
    for kinds in by_epoch.values():
        if "window" in kinds:
            continue
        if "auto" in kinds or "KL" in kinds:
            done += int("auto" in kinds) + int("KL" in kinds)
        elif "noise" in kinds:
            done += 1
    return int(done)


def curriculum_state_at_best_epoch(loss_log, best_epoch, defaults) -> dict:
    """Curriculum position at the start of the epoch after ``best_epoch``.

    Warm-up series store the value used during that epoch, before a bump.
    Bump rows recorded on ``best_epoch`` have already been applied, so
    their ``new`` values override the series. Patience is the post-bump
    ``cpt_patience_epochs`` entry. ``auto_kl_done`` counts auto/KL bumps
    that have already fired, including one on the best epoch itself.
    """
    defaults = dict(defaults)
    index = int(best_epoch)
    state = {
        "current_auto_warm": float(
            _series_at(loss_log, "auto_warm_values", index, defaults["auto"])
        ),
        "current_kl_warm": float(
            _series_at(loss_log, "kl_warm_values", index, defaults["kl"])
        ),
        "current_noise_warm": float(
            _series_at(loss_log, "noise_warm_values", index, defaults["noise"])
        ),
        "current_sequence_len": int(
            round(
                _series_at(
                    loss_log,
                    "sequence_len_values",
                    index,
                    defaults["sequence_len"],
                )
            )
        ),
        "cpt_patience": int(
            round(
                _series_at(
                    loss_log,
                    "cpt_patience_epochs",
                    index,
                    defaults.get("cpt_patience", 0),
                )
            )
        ),
        "warm_up_happened": False,
        "auto_kl_done": 0,
    }
    events = []
    if isinstance(loss_log, dict) and isinstance(loss_log.get("bump_events"), list):
        events = loss_log["bump_events"]
    state["auto_kl_done"] = auto_kl_done_through_epoch(events, index)
    for row in events:
        if not isinstance(row, dict):
            continue
        if int(row.get("epoch", -1)) != index:
            continue
        state["warm_up_happened"] = True
        kind = str(row.get("type", ""))
        new = row.get("new")
        if new is None:
            continue
        if kind == "auto":
            state["current_auto_warm"] = float(new)
        elif kind == "KL":
            state["current_kl_warm"] = float(new)
        elif kind == "noise":
            state["current_noise_warm"] = float(new)
        elif kind == "window":
            state["current_sequence_len"] = int(round(float(new)))
    return state
