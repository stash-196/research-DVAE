"""Lorenz63 observation slices: only_x, only_xy, only_xz, all_xyz."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from dvae.dataset.lorenz63_dataset import (
    LORENZ63_SLICE_CHANNELS,
    Lorenz63,
    lorenz63_resolved_x_dim,
)
from dvae.eval.utils.benchmark_signals import resolve_channel_keys
from dvae.eval.utils.batch_all_visuals import resolve_reference_channel_spec


def _apply(process: str, sequence: np.ndarray) -> torch.Tensor:
    dataset = Lorenz63.__new__(Lorenz63)
    dataset.observation_process = process
    return dataset.apply_observation_process(sequence)


def _xyz(n: int = 4) -> np.ndarray:
    """Tiny xyz series with distinct columns so a wrong slice fails."""
    t = np.arange(n, dtype=np.float64)
    return np.stack([t, t + 100.0, t + 1000.0], axis=1)


def test_only_x_slice_shape_and_values():
    xyz = _xyz()
    out = _apply("only_x", xyz)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (xyz.shape[0],)
    assert torch.equal(out, torch.tensor(xyz[:, 0], dtype=torch.float32))


@pytest.mark.parametrize(
    "process, columns",
    [
        ("only_xy", [0, 1]),
        ("only_xz", [0, 2]),
        ("all_xyz", [0, 1, 2]),
    ],
)
def test_new_slice_shapes_and_columns(process, columns):
    xyz = _xyz()
    out = _apply(process, xyz)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (xyz.shape[0], len(columns))
    expected = torch.tensor(xyz[:, columns], dtype=torch.float32)
    assert torch.equal(out, expected)


def test_xyz_to_xyz_unchanged():
    xyz = _xyz()
    out = _apply("xyz_to_xyz", xyz)
    assert out.shape == (xyz.shape[0], 3)
    assert torch.equal(out, torch.tensor(xyz, dtype=torch.float32))


def test_unknown_observation_process_raises():
    with pytest.raises(ValueError, match="not recognized"):
        _apply("only_yz", _xyz())


def test_resolved_x_dim_leaves_only_x_windowing_alone():
    assert lorenz63_resolved_x_dim("only_x", 1) == 1
    assert lorenz63_resolved_x_dim("only_x", 4) == 4
    assert lorenz63_resolved_x_dim("only_x", None) is None
    assert lorenz63_resolved_x_dim("only_xy", 1) == 2
    assert lorenz63_resolved_x_dim("only_xz", None) == 2
    assert lorenz63_resolved_x_dim("all_xyz", 1) == 3
    assert lorenz63_resolved_x_dim("xyz_to_xyz", 3) == 3
    assert lorenz63_resolved_x_dim("only_x_indicate", 1) == 1


def test_channel_keys_follow_slice_names():
    assert resolve_channel_keys("only_x", 1, "Lorenz63") == [("x", 0)]
    assert resolve_channel_keys("only_xy", 2, "Lorenz63") == [("x", 0), ("y", 1)]
    assert resolve_channel_keys("only_xz", 2, "Lorenz63") == [("x", 0), ("z", 1)]
    assert resolve_channel_keys("all_xyz", 3, "Lorenz63") == [
        ("x", 0),
        ("y", 1),
        ("z", 2),
    ]
    # xyz_to_xyz is not a slice alias; keep the generic dim names.
    assert resolve_channel_keys("xyz_to_xyz", 3, "Lorenz63") == [
        ("dim0", 0),
        ("dim1", 1),
        ("dim2", 2),
    ]
    # only_x windowed with x_dim > 1 is still not named y/z.
    assert resolve_channel_keys("only_x", 4, "Lorenz63") == [
        ("dim0", 0),
        ("dim1", 1),
        ("dim2", 2),
        ("dim3", 3),
    ]
    assert LORENZ63_SLICE_CHANNELS["all_xyz"] == ("x", "y", "z")


def test_batch_all_lorenz_reference_stays_full_xyz():
    spec = resolve_reference_channel_spec("Lorenz63", observation_process="only_xz")
    assert spec["primary_name"] == "x"
    assert spec["column_names"] == ["x", "y", "z"]
    xhro = resolve_reference_channel_spec("Xhro", observation_process="only_x")
    assert xhro["primary_name"] == "ch4"


def _write_lorenz_pickle(root: Path, xyz: np.ndarray, label: str = "tiny") -> None:
    folder = root / "lorenz63" / "data" / label
    folder.mkdir(parents=True)
    for split in ("train", "test"):
        with open(folder / f"complete_dataset_{split}.pkl", "wb") as handle:
            pickle.dump(xyz, handle)


def _dataset(root: Path, **overrides) -> Lorenz63:
    params = dict(
        data_dir=str(root),
        dataset_label="tiny",
        mask_label="None",
        split="train",
        seq_len=8,
        x_dim=1,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.25,
        observation_process="only_x",
        device="cpu",
        overlap=False,
        with_nan=False,
        shuffle=False,
    )
    params.update(overrides)
    return Lorenz63(**params)


def test_dataset_only_x_item_is_column_x(tmp_path):
    xyz = _xyz(32)
    _write_lorenz_pickle(tmp_path, xyz)
    dataset = _dataset(tmp_path, observation_process="only_x", x_dim=1)
    assert dataset.x_dim == 1
    item = np.asarray(dataset[0])
    assert item.shape == (8, 1)
    assert np.allclose(item[:, 0], xyz[:8, 0])


@pytest.mark.parametrize(
    "process, columns, width",
    [
        ("only_xy", [0, 1], 2),
        ("only_xz", [0, 2], 2),
        ("all_xyz", [0, 1, 2], 3),
    ],
)
def test_dataset_construction_sets_width(tmp_path, process, columns, width):
    xyz = _xyz(32)
    _write_lorenz_pickle(tmp_path, xyz)
    # Pass the historical only_x default so resolution, not the caller, sets width.
    dataset = _dataset(tmp_path, observation_process=process, x_dim=1)
    assert dataset.x_dim == width
    item = np.asarray(dataset[0])
    assert item.shape == (8, width)
    assert np.allclose(item, xyz[:8, columns])


def test_only_x_windowing_still_uses_caller_x_dim(tmp_path):
    xyz = _xyz(32)
    _write_lorenz_pickle(tmp_path, xyz)
    dataset = _dataset(tmp_path, observation_process="only_x", x_dim=4, seq_len=4)
    assert dataset.x_dim == 4
    item = np.asarray(dataset[0])
    assert item.shape == (4, 4)
    assert np.allclose(item.reshape(-1), xyz[:16, 0])


def test_build_dataloader_overrides_multichannel_x_dim(tmp_path):
    xyz = _xyz(32)
    _write_lorenz_pickle(tmp_path, xyz)
    from dvae.dataset.dataset_builder import DatasetConfig, build_dataloader

    cfg = DatasetConfig(
        data_dir=str(tmp_path),
        x_dim=1,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.25,
        observation_process="only_xy",
        overlap=False,
        with_nan=False,
        seq_len=8,
        device="cpu",
        dataset_label="tiny",
        mask_label="None",
    )
    train_loader, _val_loader, n_train, n_val = build_dataloader(
        "Lorenz63", cfg, "train"
    )
    assert cfg.x_dim == 2
    assert n_train > 0 and n_val > 0
    batch = next(iter(train_loader))
    assert tuple(batch.shape) == (min(2, n_train), 8, 2)
    assert torch.allclose(batch[0], torch.tensor(xyz[:8, [0, 1]], dtype=torch.float32))
