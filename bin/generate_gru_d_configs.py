#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write GRU-D ini files. Does not touch the OTF config generator.

Outputs land under the directory you pass, intended to be
``config/general_signal/generated/<experiment>/`` (gitignored). Existing
hand-written configs, including ``config/lorenz63`` and
``bin/generate_config_file_for_multiple.py``, are not modified.

Edit the sweep lists in ``main`` on Deigo, then:

    python3 bin/generate_gru_d_configs.py
    bash scripts/slurm/train/run_gru_d_multiple.sh <experiment_name>
"""

from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "config" / "general_signal" / "cfg_gru_d_template.ini"

# State width GRU-D should see. These are coordinates, not delay windows.
OBSERVATION_X_DIM = {
    "only_x": 1,
    "only_xy": 2,
    "only_xz": 2,
    "all_xyz": 3,
    "xyz_to_xyz": 3,
}


def assert_output_dir_allowed(output_dir: Path):
    parts = Path(output_dir).resolve().parts
    if "lorenz63" in parts and "generated" not in parts:
        raise RuntimeError(
            f"refusing to write GRU-D configs into {output_dir}. "
            "Use config/general_signal/generated/<experiment>."
        )


def _as_text(value):
    if isinstance(value, (list, tuple)):
        return ",".join(str(piece) for piece in value)
    return str(value)


def generate_gru_d_configs(output_dir, experiment_name, sweeps, template_path=None):
    """Expand ``sweeps`` (dict of lists) into GRU-D ini files.

    Returns the written paths. Only files named ``cfg_gru_d_*.ini`` are
    created, and only inside ``output_dir``.
    """
    output_dir = Path(output_dir)
    assert_output_dir_allowed(output_dir)
    template_path = Path(template_path) if template_path else TEMPLATE
    template = template_path.read_text()
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = list(sweeps.keys())
    written = []
    for values in itertools.product(*[sweeps[key] for key in keys]):
        combo = dict(zip(keys, values))
        obs = combo["observation_process"]
        if obs not in OBSERVATION_X_DIM:
            raise ValueError(
                f"observation_process {obs!r} has no GRU-D width in "
                f"{sorted(OBSERVATION_X_DIM)}. Add it explicitly; do not "
                "reuse a delay-window x_dim."
            )
        combo["x_dim"] = OBSERVATION_X_DIM[obs]
        combo["experiment_name"] = experiment_name
        combo.setdefault("dataset_name", "Lorenz63")
        combo.setdefault("dataset_label", "sigma10_rho28_beta8d3_N108k_dt0.01")
        combo.setdefault("tag", "GRU_D")
        mapping = {key: _as_text(value) for key, value in combo.items()}
        mapping["test_keys"] = ",".join(keys)
        text = template.format_map(defaultdict(str, mapping))
        # Short, unique names. A full cartesian label exceeds typical path limits.
        digest = hashlib.sha1(
            repr(tuple(sorted((key, _as_text(combo[key])) for key in keys))).encode()
        ).hexdigest()[:8]
        filename = (
            "cfg_gru_d_{exp}_{obs}_mask-{mask}_h{hidden}_T{seq}_ep{epochs}_{digest}.ini"
        ).format(
            exp=experiment_name,
            obs=_as_text(combo["observation_process"]),
            mask=_as_text(combo["mask_label"]),
            hidden=_as_text(combo["dim_rnn"]),
            seq=_as_text(combo["sequence_len"]),
            epochs=_as_text(combo["epochs"]),
            digest=digest,
        ).replace(" ", "")
        if not filename.startswith("cfg_gru_d_") or not filename.endswith(".ini"):
            raise RuntimeError(f"unexpected config name {filename}")
        path = output_dir / filename
        if path.parent.resolve() != output_dir.resolve():
            raise RuntimeError(f"refusing to write outside {output_dir}: {path}")
        path.write_text(text)
        written.append(path)
    return written


def default_sweeps():
    """Lorenz comparison grid. Edit this on Deigo; leave the OTF generator alone.

    ``mask_label`` must match a mask pickle already on disk
    (``mask_<distribution>_pnan<rate>_{train,test}.pkl``). ``None`` is fully
    observed. Epoch count should be set beside the OTF runs you want to
    compare against; OTF grids in this repo are often much longer than 2000.
    """
    return {
        "observation_process": ["all_xyz", "only_x"],
        "mask_label": ["None"],
        "dim_rnn": [64],
        "dropout_p": [0.0],
        "lr": [0.001],
        "epochs": [2000],
        "early_stop_patience": [50],
        "save_frequency": [50],
        "gradient_clip": [1.0],
        "seed": [0],
        "dataset_name": ["Lorenz63"],
        "dataset_label": ["sigma10_rho28_beta8d3_N108k_dt0.01"],
        "shuffle": ["True"],
        "batch_size": [128],
        "num_workers": [4],
        "sequence_len": [400],
        "val_indices": [0.2],
    }


def main():
    experiment_name = "gru_d_lorenz_baselines"
    output_dir = (
        REPO_ROOT / "config" / "general_signal" / "generated" / experiment_name
    )
    written = generate_gru_d_configs(output_dir, experiment_name, default_sweeps())
    print(f"Wrote {len(written)} GRU-D configs to {output_dir}")
    print("Submit:")
    print(f"  bash scripts/slurm/train/run_gru_d_multiple.sh {experiment_name}")


if __name__ == "__main__":
    main()
