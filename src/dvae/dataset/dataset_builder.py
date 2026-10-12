# dataset_builder.py
from torch.utils.data import DataLoader
from dataclasses import asdict

# Import dataset classes
from dvae.dataset.lorenz63_dataset import Lorenz63, lorenz63_resolved_x_dim
from dvae.dataset.sho_dataset import SimpleHarmonicOscillator
from dvae.dataset.damped_sho_dataset import DampedSimpleHarmonicOscillator
from dvae.dataset.xhro_dataset import Xhro
from dvae.dataset.xhro_packet_loss_dataset import XhroPacketLoss
from dvae.dataset.physionet2012_dataset import PhysioNet2012
from dataclasses import dataclass

from dvae.dataset.xhro_proper_dataset import XhroProper

DATASET_REGISTRY = {
    "Lorenz63": Lorenz63,
    "Xhro": Xhro,
    "XhroPacketLoss": XhroPacketLoss,
    "PhysioNet2012": PhysioNet2012,
    "SHO": SimpleHarmonicOscillator,
    "DampedSHO": DampedSimpleHarmonicOscillator,
}
if XhroProper is not None:
    DATASET_REGISTRY["XhroProper"] = XhroProper


@dataclass
class DatasetConfig:
    data_dir: str
    x_dim: int
    batch_size: int
    shuffle: bool
    num_workers: int
    sample_rate: int
    skip_rate: int
    val_indices: float
    observation_process: str
    overlap: bool
    with_nan: bool
    seq_len: int
    device: str
    data_dir: str
    dataset_label: str | None = None
    mask_label: str | None = None
    # XhroProper session parent. Other datasets accept and ignore these.
    # None keeps XhroProper's packet-loss grok_output path.
    data_root: str | None = None
    corpus: str | None = None
    # XhroProper coarse modes. None keeps usable_tail and artifact mask off.
    coarse_test_split: str | None = None
    coarse_eeg_artifact_mask: str | None = None
    coarse_eeg_artifact_k: float | None = None
    coarse_eeg_max_rel_delta: float | None = None
    # HR coarse columns (hr_mean_bpm) stay off unless this is true.
    coarse_include_hr: bool = False
    # XhroProper waveform crop, seconds from the first timestamp. None = off.
    crop_start_sec: float | None = None
    crop_end_sec: float | None = None
    # XhroProper waveform. None keeps filled gaps inside windows.
    # filled_nan_log_sec is the re-cut log cutoff when max_fill is unset.
    max_fill_gap_sec: float | None = None
    filled_nan_log_sec: float | None = None
    # Window rules. Blank config keys take these. legacy_defaults
    # (alias legacy_windowing) true restores the previous bundle when the
    # placement keys are blank, and turns the NaN filter off when
    # max_window_nan_ratio is blank.
    # A blank ratio (None, and not max_window_nan_ratio_off) uses
    # dataset_missing_rate + max_window_nan_excess.
    # Both names are stored. Either one true turns the legacy bundle on,
    # and __post_init__ writes that same flag into both fields so
    # windowing_from_config can be unpacked here.
    legacy_windowing: bool = False
    legacy_defaults: bool = False
    max_gap_window_frac: float | None = 0.25
    # None disables the quantile term. 0.99 is the blank-key default.
    gap_split_quantile: float | None = 0.99
    start_on_observed: bool = True
    strict_non_overlap: bool = True
    max_window_nan_ratio: float | None = None
    max_window_nan_ratio_off: bool = False
    max_window_nan_excess: float | None = 0.2
    # Waveform block mask. None file means no masking. Mode None is not_good.
    block_mask_file: str | None = None
    block_mask_mode: str | None = None

    def __post_init__(self):
        # ``legacy_defaults`` and ``legacy_windowing`` are one switch.
        # windowing_from_config returns both, already OR-ed. A caller that
        # sets only one name still lands on the same flag.
        legacy = bool(self.legacy_windowing) or bool(self.legacy_defaults)
        self.legacy_windowing = legacy
        self.legacy_defaults = legacy


def build_dataloader(
    dataset_name: str,
    dataset_config: DatasetConfig,
    split: str,
    eval_mode: bool = False,
):
    """Unified function to build data loaders for all datasets.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'Lorenz63', 'Xhro').
        dataset_config (DatasetConfig): Config object with dataset parameters.

    Returns:
        tuple: (train_dataloader, val_dataloader, train_num, val_num)
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset_name == "Lorenz63":
        dataset_config.x_dim = lorenz63_resolved_x_dim(
            dataset_config.observation_process, dataset_config.x_dim
        )

    dataset_class = DATASET_REGISTRY[dataset_name]
    dataset_params = asdict(dataset_config)  # Convert DatasetConfig to dict

    if split == "train":
        # Load train and validation datasets for training
        train_dataset = dataset_class(split="train", **dataset_params)
        val_dataset = dataset_class(split="valid", **dataset_params)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=dataset_config.batch_size,
            shuffle=dataset_config.shuffle,
            num_workers=dataset_config.num_workers,
            pin_memory=(dataset_config.device in ["cuda", "mps"]),
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=dataset_config.batch_size,
            shuffle=dataset_config.shuffle,
            num_workers=dataset_config.num_workers,
            pin_memory=(dataset_config.device in ["cuda", "mps"]),
        )
        return train_dataloader, val_dataloader, len(train_dataset), len(val_dataset)

    elif split == "test":
        # Load test dataset for evaluation
        test_dataset = dataset_class(
            split="test", eval_mode=eval_mode, **dataset_params
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=dataset_config.batch_size,
            shuffle=False,  # No shuffling for evaluation
            num_workers=dataset_config.num_workers,
            pin_memory=(dataset_config.device in ["cuda", "mps"]),
        )
        return test_dataloader

    else:
        raise ValueError(f"Unsupported split: {split}")
