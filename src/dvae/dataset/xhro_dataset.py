#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" """

import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from .gap_grid import (
    PRESENT_COL,
    apply_nan_window_kwargs,
    dataset_missing_rate,
    filter_window_starts,
    log_current_nan_window_rule,
    log_gap_split_recut,
    log_window_recut,
    observed_row_mask,
    resolve_recut_length,
    shift_starts_onto_observed,
)
from .utils import data_utils
from .utils.visualize_nans import plot_nan_heatmap, plot_segment_length_hist


def _resolve_original_observation_process(observation_process):
    """Return the base original observation key for exact or suffixed variants."""
    if observation_process in select_columns_for_obs_conditions["original"]:
        return observation_process

    if isinstance(observation_process, str):
        for suffix in ("_interpolate", "_indicate"):
            if observation_process.endswith(suffix):
                base = observation_process[: -len(suffix)]
                if base in select_columns_for_obs_conditions["original"]:
                    return base

    return None


class Xhro(Dataset):
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
        """
        :param path_to_data: path to the data folder
        :param dataset_label: label for the dataset
        :param split: 'train' or 'valid'
        :param seq_len: length of the sequences
        :param x_dim: dimension of the data
        :param sample_rate: downsampling rate
        :param skip_rate: skip rate for data sampling
        :param val_indices: proportion of data used for validation
        :param observation_process: process to apply to observations
        :param device: computation device
        :param overlap: whether to use overlapping sequences
        :param shuffle: whether to shuffle the data indices
        """

        self.path_to_data = data_dir
        self.dataset_label = dataset_label
        self.mask_label = mask_label
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
        # Optional window rules. XhroProper sets the parsed values; a plain
        # Xhro run stores whatever the config passed and leaves them off.
        self.start_on_observed = bool(kwargs.get("start_on_observed", True))
        self.strict_non_overlap = bool(kwargs.get("strict_non_overlap", True))
        self.max_gap_window_frac = kwargs.get("max_gap_window_frac", 0.25)
        # Blank ratio uses dataset_missing_rate + excess (default 0.2).
        apply_nan_window_kwargs(self, kwargs)
        self.dataset_missing_rate = None
        self.filled_gap_spans = kwargs.get("filled_gap_spans", None)
        self.observed_per_row = kwargs.get("observed_per_row", None)

        original_base = _resolve_original_observation_process(self.observation_process)

        # Read data from file
        if original_base is not None:
            filename = f"{self.path_to_data}/xhro/processed/{self.dataset_label}/filtered_data.parquet"
            the_sequence = pd.read_parquet(filename)
            self.sampling_freq = 250
            print(f"Downloaded data from {filename}")
        elif (
            self.observation_process
            in select_columns_for_obs_conditions["coarsed"].keys()
        ):
            filename = os.path.join(
                f"{self.path_to_data}",
                "xhro",
                "processed",
                f"{self.dataset_label}",
                "coarse_features.parquet",
            )
            the_sequence = pd.read_parquet(filename)
            self.sampling_freq = 1 / (60 * 5)
            print(f"Downloaded data from {filename}")
        else:
            raise ValueError(
                f"Invalid observation process: {self.observation_process}. "
                "Must be one of the keys in select_columns_for_obs_conditions."
            )
        # see if directory of filename exists
        the_sequence = select_observation_window_slice(self.dataset_label, the_sequence)
        # Once, on the whole loaded series, before the train/test cut and
        # before observation processing hides real gaps.
        self.dataset_missing_rate = self._missing_rate(the_sequence)

        if self.split == "test":  # use the Latter 20% of the data for testing
            the_sequence = the_sequence[-the_sequence.shape[0] // 5 :]
        else:
            the_sequence = the_sequence[: -the_sequence.shape[0] // 5]

        # Store the full sequence before applying any observation process
        self.full_sequence = the_sequence
        # Extract and store the missing mask before applying observation process
        self.missing_mask = self._extract_missing_mask(the_sequence)
        # Process the sequence based on the observation process
        the_sequence = self.apply_observation_process(the_sequence)

        # the_sequence should be squeezed before this takes place
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

        # Generate sequences with or without overlap
        self.is_segmented_1d = False
        if the_sequence.ndim == 1:
            if self.x_dim > 1:
                self.is_segmented_1d = True
            if self.overlap:
                the_sequence = self.create_moving_window_sequences(
                    the_sequence, self.x_dim
                )
            else:  # Remove the last sequence if it is not the correct length
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
        # Now the_sequence is a 2D tensor
        self.seq = the_sequence
        # torch.Size([80000, 1])

        self.update_sequence_length(self.seq_len)

    def _values_for_missing_rate(self, sequence):
        """Observation columns only, with no indicate channel and no trim."""
        process = self.observation_process
        cols = None
        original = select_columns_for_obs_conditions["original"]
        coarsed = select_columns_for_obs_conditions["coarsed"]
        if process in original:
            cols = original[process]
        else:
            base = _resolve_original_observation_process(process)
            if base is not None:
                cols = original[base]
            elif process in coarsed:
                cols = coarsed[process]
        if cols is not None:
            values = sequence[list(cols)].to_numpy(dtype=np.float64)
        elif process in {"only_x_interpolate", "only_x_indicate"}:
            values = sequence.iloc[:, 0].to_numpy(dtype=np.float64)
        else:
            values = sequence.select_dtypes(include="number").to_numpy(dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return values

    def _missing_rate(self, sequence) -> float:
        present = None
        columns = getattr(sequence, "columns", [])
        if PRESENT_COL in columns:
            present = sequence[PRESENT_COL].to_numpy(dtype=bool)
        return dataset_missing_rate(self._values_for_missing_rate(sequence), present)

    def _extract_missing_mask(self, sequence):
        """
        Extracts a boolean mask indicating which values are missing (NaN) in the input sequence.
        This is called before observation_process so the mask is tied to the original data.

        Returns:
            np.ndarray: Boolean mask where True indicates a missing value (NaN).
        """
        if (
            self.observation_process
            in select_columns_for_obs_conditions["original"].keys()
        ):
            first_valid_index = sequence.first_valid_index()
            if first_valid_index is None:
                values = sequence[
                    select_columns_for_obs_conditions["original"][
                        self.observation_process
                    ]
                ].to_numpy(dtype=np.float64)
            else:
                values = (
                    sequence[
                        select_columns_for_obs_conditions["original"][
                            self.observation_process
                        ]
                    ]
                    .loc[first_valid_index:]
                    .to_numpy(dtype=np.float64)
                )
            return np.isnan(values)

        original_base = _resolve_original_observation_process(self.observation_process)
        if original_base is not None:
            first_valid_index = sequence.first_valid_index()
            values = sequence[
                select_columns_for_obs_conditions["original"][original_base]
            ]
            if first_valid_index is not None:
                values = values.loc[first_valid_index:]
            return np.isnan(values.to_numpy(dtype=np.float64))

        if (
            self.observation_process
            in select_columns_for_obs_conditions["coarsed"].keys()
        ):
            values = sequence[
                select_columns_for_obs_conditions["coarsed"][self.observation_process]
            ].to_numpy(dtype=np.float64)
            return np.isnan(values)

        if self.observation_process in {"only_x_interpolate", "only_x_indicate"}:
            values = sequence.iloc[:, 0].to_numpy(dtype=np.float64)
            return np.isnan(values)

        return np.isnan(sequence)

    def apply_observation_process(self, sequence) -> torch.Tensor:
        """
        Applies an observation process to the sequence data.
        """
        if (
            self.observation_process
            in select_columns_for_obs_conditions["coarsed"].keys()
        ):
            data = sequence[
                select_columns_for_obs_conditions["coarsed"][self.observation_process]
            ].to_numpy(dtype=np.float32)
            normalized_data = self.normalize(data)
            return torch.from_numpy(normalized_data)
        elif (
            self.observation_process
            in select_columns_for_obs_conditions["original"].keys()
        ):
            # Find the first non-NaN index
            first_valid_index = sequence.first_valid_index()
            data = (
                sequence[
                    select_columns_for_obs_conditions["original"][
                        self.observation_process
                    ]
                ]
                .loc[first_valid_index:]
                .to_numpy(dtype=np.float32)
            )
            normalized_data = self.normalize(data)
            return torch.tensor(normalized_data, dtype=torch.float32)
        # Support for pattern: <base>_interpolate where <base> is an "original" key
        elif isinstance(
            self.observation_process, str
        ) and self.observation_process.endswith("_interpolate"):
            base = self.observation_process.rsplit("_", 1)[0]
            if base in select_columns_for_obs_conditions["original"].keys():
                # Reuse the same trimming logic as the `original` branch
                first_valid_index = sequence.first_valid_index()
                data = sequence[select_columns_for_obs_conditions["original"][base]]
                if first_valid_index is not None:
                    data = data.loc[first_valid_index:]
                arr = data.to_numpy(dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                n_nan_before = int(np.isnan(arr).sum())
                idx = np.arange(arr.shape[0])
                for j in range(arr.shape[1]):
                    x = arr[:, j]
                    nan_mask = np.isnan(x)
                    if nan_mask.any():
                        valid = ~nan_mask
                        if valid.any():
                            x[nan_mask] = np.interp(idx[nan_mask], idx[valid], x[valid])
                        else:
                            x[:] = 0.0
                    arr[:, j] = x
                n_nan_after = int(np.isnan(arr).sum())
                normalized = self.normalize(arr)
                print(
                    f"[Xhro][{self.observation_process}] NaNs before: {n_nan_before}, "
                    f"after: {n_nan_after}, shape: {normalized.shape}"
                )
                out = normalized[:, 0] if normalized.shape[1] == 1 else normalized
                return torch.tensor(out, dtype=torch.float32)
            raise ValueError(f"Invalid observation process: {self.observation_process}")

        # Support for pattern: <base>_indicate where <base> is an "original" key.
        # Layout: interleaved [x0, m0, x1, m1, ...] so x_dim = 2 * n_channels.
        elif isinstance(
            self.observation_process, str
        ) and self.observation_process.endswith("_indicate"):
            base = self.observation_process.rsplit("_", 1)[0]
            if base in select_columns_for_obs_conditions["original"].keys():
                # Reuse trimming logic from the `original` branch
                first_valid_index = sequence.first_valid_index()
                data = sequence[select_columns_for_obs_conditions["original"][base]]
                if first_valid_index is not None:
                    data = data.loc[first_valid_index:]
                arr = data.to_numpy(dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                missing_mask = np.isnan(arr)
                missing_count = int(missing_mask.sum())
                # Normalize while NaNs are present so stats ignore missing values
                x_normalized = self.normalize(arr)
                x_imputed = np.nan_to_num(x_normalized, nan=0.0).astype(np.float32)
                is_observed = (~missing_mask).astype(np.float32)
                # Interleave signal and mask per channel: [x0, m0, x1, m1, ...]
                t, c = x_imputed.shape
                result = np.empty((t, 2 * c), dtype=np.float32)
                result[:, 0::2] = x_imputed
                result[:, 1::2] = is_observed
                print(
                    f"[Xhro][{self.observation_process}] missing_count: {missing_count}, "
                    f"result_shape: {result.shape}"
                )
                return torch.tensor(result, dtype=torch.float32)
            raise ValueError(f"Invalid observation process: {self.observation_process}")
        elif self.observation_process == "only_x_interpolate":
            # take only x (first dimension) and linearly interpolate NaNs
            x = sequence.iloc[:, 0].to_numpy(dtype=np.float64)
            nan_mask = np.isnan(x)
            n_nan_before = int(np.sum(nan_mask))
            if n_nan_before > 0:
                idx = np.arange(x.shape[0])
                valid = ~nan_mask
                if valid.any():
                    x[nan_mask] = np.interp(idx[nan_mask], idx[valid], x[valid])
                else:
                    x[:] = 0.0
            n_nan_after = int(np.isnan(x).sum())
            normalized_x = self.normalize(x.reshape(-1, 1))[:, 0]
            print(
                f"[Xhro][only_x_interpolate] NaNs before: {n_nan_before}, after: {n_nan_after}"
            )
            return torch.tensor(normalized_x, dtype=torch.float32)
        elif self.observation_process == "only_x_indicate":
            # produce two-dimensional observation: [x_imputed, is_observed]
            x = sequence.iloc[:, 0].to_numpy(dtype=np.float32)
            missing_mask = np.isnan(x)
            missing_count = int(np.sum(missing_mask))
            x_imputed = np.nan_to_num(x, nan=0.0).astype(np.float32)
            x_normalized = self.normalize(x_imputed.reshape(-1, 1))[:, 0]
            is_observed = (~missing_mask).astype(np.float32)
            result = np.stack([x_normalized, is_observed], axis=1)
            print(
                f"[Xhro][only_x_indicate] missing_count: {missing_count}, result_shape: {result.shape}"
            )
            return torch.tensor(result, dtype=torch.float32)
        else:
            raise ValueError(f"Invalid observation process: {self.observation_process}")

    def normalize(self, data):
        """
        Normalize the data.
        """
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        data = (data - mean) / std
        return data

    def get_full_xyz(self, index):
        """
        Retrieves the full xyz variables for the given index.

        Args:
            index (int): Index of the desired data sequence.

        Returns:
            torch.Tensor: The full xyz sequence data for the given index.
        """
        start_frame = self.data_idx[index]
        end_frame = min(start_frame + self.seq_len, len(self.full_sequence))
        # Return the full (x, y, z) data
        return self.full_sequence[start_frame:end_frame]

    def get_missing_mask(self, index):
        """
        Retrieves the missing value mask for the given index.
        Indicates which values were originally NaN before imputation/interpolation.

        Args:
            index (int): Index of the desired data sequence.

        Returns:
            np.ndarray: Boolean mask where True indicates a missing value.
        """
        start_frame = self.data_idx[index]
        end_frame = min(start_frame + self.seq_len, len(self.missing_mask))
        return self.missing_mask[start_frame:end_frame]

    @staticmethod
    def create_moving_window_sequences(sequence, window_size):
        """
        Converts a time series into a 3D array of overlapping sequences.
        """
        num_samples = sequence.shape[0]
        stride = sequence.strides[0]
        num_sequences = num_samples - window_size + 1
        shape = (num_sequences, window_size, sequence.shape[1])
        strides = (stride, stride, sequence.strides[1])
        return np.lib.stride_tricks.as_strided(sequence, shape=shape, strides=strides)

    def split_dataset(self, indices, val_ratio):
        """
        Splits the dataset indices into training and validation sets.
        """
        if self.shuffle:
            # Fixed local RNG: same split as the old global seed 1234567890, without touching np.random.
            np.random.RandomState(1234567890).shuffle(indices)
        split_point = int(len(indices) * (1 - val_ratio))
        return indices[:split_point], indices[split_point:]

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, index):
        start_frame = self.data_idx[index]
        end_frame = min(start_frame + self.seq_len, len(self.seq))
        return self.seq[start_frame:end_frame]

    def _window_starts(self, starts, boundaries):
        """Honor ``start_on_observed``. Off leaves the starts unchanged."""
        if not getattr(self, "start_on_observed", False):
            return starts
        observed = observed_row_mask(
            self.seq, getattr(self, "observed_per_row", None)
        )
        return shift_starts_onto_observed(
            starts,
            observed,
            self.seq_len,
            boundaries,
            int(self.seq.shape[0]),
        )

    def update_sequence_length(self, new_seq_len=None, minimum_nan_ratio=None):
        # Only one index representing the start of each sequence.
        # ``split_boundaries`` (sample indices) are kept on the dataset and
        # applied on every re-cut, including curriculum resizes.
        # ``max_gap_window_frac`` adds cuts for this length only.
        if new_seq_len is not None:
            num_frames = self.seq.shape[0]
            requested = int(new_seq_len)
            spans = getattr(self, "missing_run_spans", None)
            if spans is None:
                spans = getattr(self, "filled_gap_spans", None)
            seq_len, boundaries = resolve_recut_length(
                num_frames,
                requested,
                getattr(self, "split_boundaries", None),
                spans,
                getattr(self, "max_gap_window_frac", None),
                gap_quantile=getattr(self, "dataset_gap_quantile", None),
            )
            if boundaries is not None and seq_len < requested:
                print(
                    f"[{type(self).__name__}] seq_len {requested} exceeds the "
                    f"longest unbroken span {seq_len}; using {seq_len}."
                )
            self.seq_len = seq_len
            log_gap_split_recut(self, self.seq_len, num_frames)
            # Recalculate data_idx based on the new sequence length.
            # Each span uses the local RandomState inside find_indices.
            strict = bool(getattr(self, "strict_non_overlap", False))
            observed = None
            if strict and getattr(self, "start_on_observed", False):
                observed = observed_row_mask(
                    self.seq, getattr(self, "observed_per_row", None)
                )
            all_indices = data_utils.find_indices(
                num_frames,
                self.seq_len,
                num_frames // self.seq_len,
                boundaries=boundaries,
                strict_non_overlap=strict,
                observed=observed,
            )
            all_indices = self._window_starts(all_indices, boundaries)
            valid_frames = filter_window_starts(self, all_indices, minimum_nan_ratio)

            train_indices, validation_indices = self.split_dataset(
                valid_frames, self.val_indices
            )
            if self.split == "train":
                valid_frames = train_indices
            else:
                valid_frames = validation_indices
            self.data_idx = list(valid_frames)
        else:
            # Use entire sequence if seq_len is None
            self.data_idx = [0]
            print("[Dataset] Using entire sequence (seq_len is None). Total frames: 1")
            log_gap_split_recut(self, None, getattr(self.seq, "shape", [0])[0])
            log_current_nan_window_rule(self, 0, minimum_nan_ratio)

        log_window_recut(self)
        return


select_columns_for_obs_conditions = {
    "original": {
        "raw_ch1": [
            "ch1",
        ],
        "raw_ch2": [
            "ch2",
        ],
        "raw_ch3": [
            "ch3",
        ],
        "raw_ch4": [
            "ch4",
        ],
        "raw_ecg": [
            "ch1",
            "ch2",
        ],
        "raw_eeg": [
            "ch3",
            "ch4",
        ],
        "raw_all": [
            "ch1",
            "ch2",
            "ch3",
            "ch4",
        ],
    },
    "coarsed": {
        "ch4_relative_powers": [
            ("ch4", "relative_alpha_power"),
            ("ch4", "relative_beta_power"),
            ("ch4", "relative_delta_power"),
            ("ch4", "relative_theta_power"),
            ("ch4", "relative_gamma_low_power"),
            ("ch4", "relative_gamma_mid_power"),
            ("ch4", "total_power"),
        ],
        "ch4_3_vars": [
            ("ch4", "relative_alpha_power"),
            ("ch4", "relative_beta_power"),
            ("ch4", "total_power"),
        ],
        "ch4_alpha": [("ch4", "relative_alpha_power")],
        "all_ch_relative_powers": [
            ("ch1", "relative_alpha_power"),
            ("ch1", "relative_beta_power"),
            ("ch1", "relative_delta_power"),
            ("ch1", "relative_theta_power"),
            ("ch1", "relative_gamma_low_power"),
            ("ch1", "relative_gamma_mid_power"),
            ("ch1", "total_power"),
            ("ch2", "relative_alpha_power"),
            ("ch2", "relative_beta_power"),
            ("ch2", "relative_delta_power"),
            ("ch2", "relative_theta_power"),
            ("ch2", "relative_gamma_low_power"),
            ("ch2", "relative_gamma_mid_power"),
            ("ch2", "total_power"),
            ("ch3", "relative_alpha_power"),
            ("ch3", "relative_beta_power"),
            ("ch3", "relative_delta_power"),
            ("ch3", "relative_theta_power"),
            ("ch3", "relative_gamma_low_power"),
            ("ch3", "relative_gamma_mid_power"),
            ("ch3", "total_power"),
            ("ch4", "relative_alpha_power"),
            ("ch4", "relative_beta_power"),
            ("ch4", "relative_delta_power"),
            ("ch4", "relative_theta_power"),
            ("ch4", "relative_gamma_low_power"),
            ("ch4", "relative_gamma_mid_power"),
            ("ch4", "total_power"),
        ],
    },
}


def select_observation_window_slice(dataset_exp_label, the_sequence):
    """
    Slices the data based on the dataset experiment label to select a specific time window.

    :param dataset_exp_label: The label for the dataset (e.g., "XHRO_04_XH-15").
    :param the_sequence: The pandas DataFrame with a datetime index.
    :return: The sliced DataFrame.
    """
    if dataset_exp_label == "XHRO_01_XH006":
        # Slice from Oct 6 2023 15:53 to 16:03
        start_time = pd.Timestamp("2023-10-06 15:53:00").tz_localize("Asia/Tokyo")
        end_time = pd.Timestamp("2023-10-06 16:03:00").tz_localize("Asia/Tokyo")
        # Filter rows where 'datetime' is within the range
        mask = (the_sequence["datetime"] >= start_time) & (
            the_sequence["datetime"] <= end_time
        )
        return the_sequence[mask]
    elif dataset_exp_label == "XHRO_04_XH057":
        # Slice from Nov 10 2023 14:52 to 15:02
        start_time = pd.Timestamp("2023-11-10 14:52:00").tz_localize("Asia/Tokyo")
        end_time = pd.Timestamp("2023-11-10 15:02:00").tz_localize("Asia/Tokyo")
        # Filter rows where 'datetime' is within the range
        mask = (the_sequence["datetime"] >= start_time) & (
            the_sequence["datetime"] <= end_time
        )
        return the_sequence[mask]
    elif dataset_exp_label == "XHRO_02_XH070":
        # Slice from Oct 29 2023 22:12 to 22:22
        start_time = pd.Timestamp("2023-10-29 22:12:00").tz_localize("Asia/Tokyo")
        end_time = pd.Timestamp("2023-10-29 22:22:00").tz_localize("Asia/Tokyo")
        # Filter rows where 'datetime' is within the range
        mask = (the_sequence["datetime"] >= start_time) & (
            the_sequence["datetime"] <= end_time
        )
        return the_sequence[mask]
    # Add more cases here for other labels if needed
    # elif dataset_exp_label == "another_label":
    #     start_time = pd.Timestamp('...')
    #     end_time = pd.Timestamp('...')
    #     return the_sequence.loc[start_time:end_time]
    else:
        # For other labels, return the full sequence
        return the_sequence
