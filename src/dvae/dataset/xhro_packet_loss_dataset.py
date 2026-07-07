#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset loader for xhro_packet_loss preprocessed biop parquet files."""

import numpy as np
import pandas as pd
import torch

from .xhro_dataset import Xhro, _resolve_original_observation_process


VALID_VARIANTS = ("realtime", "recovered")


def _resolve_variant(mask_label: str | None) -> str:
    """Use mask_label as the realtime/recovered variant selector."""
    if mask_label in (None, "None", ""):
        return "realtime"
    if mask_label not in VALID_VARIANTS:
        raise ValueError(
            f"mask_label must be one of {VALID_VARIANTS} for XhroPacketLoss "
            f"(variant selector), got: {mask_label!r}"
        )
    return mask_label


def _parquet_path(path_to_data: str, variant: str, recording_id: str) -> str:
    return (
        f"{path_to_data}/xhro_packet_loss/processed/"
        f"{variant}/{recording_id}/filtered_data.parquet"
    )


class XhroPacketLoss(Xhro):
    """Load packet-loss XHRO recordings from preprocessed biop parquet.

    Config mapping:
      - dataset_label: recording id (e.g. XHRO3506_20260622T142410000+0900)
      - mask_label: variant selector ('realtime' or 'recovered'; default realtime)
    """

    def __init__(
        self,
        data_dir,
        dataset_label,
        mask_label,
        split,
        seq_len,
        x_dim,
        sample_rate,
        skip_rate,
        val_indices,
        observation_process,
        device,
        overlap,
        shuffle=True,
        **kwargs,
    ):
        self.path_to_data = data_dir
        self.dataset_label = dataset_label
        self.mask_label = mask_label
        self.variant = _resolve_variant(mask_label)
        self.x_dim = x_dim
        self.seq_len = seq_len
        self.split = split
        self.sample_rate = sample_rate
        self.skip_rate = skip_rate
        self.val_indices = val_indices
        self.observation_process = observation_process
        self.overlap = overlap
        self.shuffle = shuffle
        self.device = device
        self.sampling_freq = None

        original_base = _resolve_original_observation_process(self.observation_process)
        if original_base is None:
            raise ValueError(
                f"Invalid observation process: {self.observation_process}. "
                "XhroPacketLoss v1 supports only original biop channels "
                "(raw_ch1..raw_ch4, raw_ecg, raw_eeg, raw_all and *_interpolate/*_indicate)."
            )

        filename = _parquet_path(self.path_to_data, self.variant, self.dataset_label)
        the_sequence = pd.read_parquet(filename)
        self.sampling_freq = 250
        print(f"[XhroPacketLoss][{self.variant}] Loaded data from {filename}")

        if self.split == "test":
            the_sequence = the_sequence[-the_sequence.shape[0] // 5 :]
        else:
            the_sequence = the_sequence[: -the_sequence.shape[0] // 5]

        self.full_sequence = the_sequence
        self.missing_mask = self._extract_missing_mask(the_sequence)
        the_sequence = self.apply_observation_process(the_sequence)
        the_sequence = the_sequence.squeeze()

        if self.x_dim is None:
            if the_sequence.ndim == 1:
                self.x_dim = 1
            elif the_sequence.ndim == 2:
                self.x_dim = the_sequence.shape[1]
            else:
                raise ValueError(
                    f"Expected x is {the_sequence.ndim} dimensions, got x_dim {self.x_dim} instead."
                )

        self.is_segmented_1d = False
        if the_sequence.ndim == 1:
            if self.x_dim > 1:
                self.is_segmented_1d = True
            if self.overlap:
                the_sequence = self.create_moving_window_sequences(
                    the_sequence, self.x_dim
                )
            else:
                the_sequence = np.array(
                    [
                        the_sequence[i : i + x_dim]
                        for i in range(0, len(the_sequence), x_dim)
                        if i + x_dim <= len(the_sequence)
                    ]
                )
        elif the_sequence.shape[1] != self.x_dim:
            raise ValueError(
                f"Expected x is {the_sequence.ndim} dimensions, got x_dim {self.x_dim} instead."
            )

        self.seq = the_sequence
        self.update_sequence_length(self.seq_len)