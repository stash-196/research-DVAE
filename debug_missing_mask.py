#!/usr/bin/env python3
"""
Debug script to diagnose missing_mask extraction and visualization mismatch.
This will help understand why gray background doesn't align with signal zeros.
"""

import sys

sys.path.insert(0, "src")

import torch
import numpy as np
from dvae.dataset.lorenz63_dataset import Lorenz63

# Create a simple test dataset
print("=" * 80)
print("TESTING MISSING_MASK EXTRACTION AND ALIGNMENT")
print("=" * 80)

# Create dataset with only_x_indicate observation process
dataset = Lorenz63(
    dataset_type="train",
    num_samples=10,
    sequence_length=100,
    observation_process="only_x_indicate",
    random_seed=42,
    nan_ratio=0.3,
)

print(f"\nDataset Info:")
print(f"  - Number of samples: {len(dataset)}")
print(f"  - Sequence length: {dataset.sequence_length}")
print(f"  - x_dim: {dataset.x_dim}")
print(f"  - observation_process: {dataset.observation_process}")

# Get first sample
sample = dataset[0]
print(f"\nFirst sample shape: {sample.shape}")
print(
    f"  - Dimension 0 (signal): min={sample[:, 0].min():.3f}, max={sample[:, 0].max():.3f}"
)
print(
    f"  - Dimension 1 (is_observed): min={sample[:, 1].min():.1f}, max={sample[:, 1].max():.1f}"
)

# Check missing_mask
print(f"\nMissing Mask Info:")
print(f"  - missing_mask type: {type(dataset.missing_mask)}")
print(f"  - missing_mask shape: {dataset.missing_mask.shape}")
print(f"  - missing_mask dtype: {dataset.missing_mask.dtype}")

# Verify alignment between zeros in dimension 0 and is_observed in dimension 1
sample_np = sample.numpy() if isinstance(sample, torch.Tensor) else sample
zeros_in_dim0 = np.where(sample_np[:, 0] == 0)[0]
zeros_in_dim1 = np.where(sample_np[:, 1] == 0)[0]
print(f"\nAlignment Check (First Sample):")
print(f"  - Indices where Dim0 = 0: {len(zeros_in_dim0)} points")
print(f"  - Indices where Dim1 = 0: {len(zeros_in_dim1)} points")
print(f"  - Are they the same? {np.array_equal(zeros_in_dim0, zeros_in_dim1)}")

# Check missing_mask for first sample
if hasattr(dataset, "missing_mask") and dataset.missing_mask is not None:
    sample_missing_mask = dataset.missing_mask[0]
    print(f"\n  - missing_mask[0] shape: {sample_missing_mask.shape}")
    print(f"  - missing_mask[0] dtype: {sample_missing_mask.dtype}")
    missing_indices = np.where(sample_missing_mask)[0]
    print(f"  - Indices where missing_mask[0] = True: {len(missing_indices)} points")
    print(
        f"  - missing_indices match zeros_in_dim1? {np.array_equal(missing_indices, zeros_in_dim1)}"
    )

    # Show sample values for debugging
    print(f"\nSample Values (first 20 timesteps):")
    for t in range(min(20, len(sample))):
        dim0_val = f"{sample_np[t, 0]:7.3f}"
        dim1_val = f"{sample_np[t, 1]:.0f}"
        mask_val = (
            "T"
            if sample_missing_mask[t, 0]
            else (
                "F"
                if sample_missing_mask.ndim > 1
                else ("T" if sample_missing_mask[t] else "F")
            )
        )
        match = (
            "✓"
            if (sample_np[t, 0] == 0)
            == sample_missing_mask[t if sample_missing_mask.ndim == 1 else t, 0]
            else "✗"
        )
        print(
            f"  t={t:2d}: Dim0={dim0_val}, Dim1={dim1_val}, MissingMask={mask_val} {match}"
        )

print("\n" + "=" * 80)
print("Create a batch and check extraction logic:")
print("=" * 80)

from torch.utils.data import DataLoader

# Create a dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
batch = next(iter(dataloader))
print(f"\nBatch from dataloader:")
print(f"  - batch shape: {batch.shape}")  # Should be (batch_size, seq_len, x_dim)

# Simulate the extraction logic from learning_algo.py
if hasattr(dataloader.dataset, "missing_mask"):
    seq_len_temp = batch.shape[
        0
    ]  # Note: first_batch.permute(1, 0, 2) in learning_algo flips to (seq, batch, dim)
    batch_size_temp = batch.shape[1]

    # This is what learning_algo.py does:
    batch_start_idx = (
        dataloader.dataset.data_idx[0] if len(dataloader.dataset.data_idx) > 0 else 0
    )
    batch_end_idx = min(
        batch_start_idx + seq_len_temp,
        len(dataloader.dataset.missing_mask),
    )

    print(f"\nExtraction logic (from learning_algo.py):")
    print(f"  - seq_len_temp: {seq_len_temp}")
    print(f"  - batch_size_temp: {batch_size_temp}")
    print(f"  - batch_start_idx (data_idx[0]): {batch_start_idx}")
    print(f"  - batch_end_idx: {batch_end_idx}")
    print(f"  - Extraction range: missing_mask[{batch_start_idx}:{batch_end_idx}]")

    missing_mask_slice = dataloader.dataset.missing_mask[batch_start_idx:batch_end_idx]
    print(f"  - missing_mask_slice shape: {missing_mask_slice.shape}")

    # This is the problem: data_idx[0] is the INDEX of the first sample in the batch,
    # but missing_mask is indexed sequentially by timesteps, not by sample indices.
    # So if data_idx[0] = 5 (meaning the 6th sample), but there are multiple samples,
    # the extraction is wrong because it assumes all sample sequences are stored
    # sequentially in missing_mask starting from data_idx[0].

    print(f"\n⚠️  POTENTIAL ISSUE:")
    print(f"  - data_idx[0] = {batch_start_idx} (index of first sample in batch)")
    print(f"  - But missing_mask is indexed differently!")
    print(
        f"  - If missing_mask is indexed by sample, not by time, this extraction is WRONG"
    )

print("\n" + "=" * 80)
