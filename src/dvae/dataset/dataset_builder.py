# dataset_builder.py
from torch.utils.data import DataLoader
from dataclasses import asdict

# Import dataset classes
from dvae.dataset.lorenz63_dataset import Lorenz63
from dvae.dataset.xhro_dataset import Xhro
from dataclasses import dataclass

DATASET_REGISTRY = {
    "Lorenz63": Lorenz63,
    "Xhro": Xhro,
}


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


def build_dataloader(dataset_name: str, dataset_config: DatasetConfig, split: str):
    """Unified function to build data loaders for all datasets.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'Lorenz63', 'Xhro').
        dataset_config (DatasetConfig): Config object with dataset parameters.

    Returns:
        tuple: (train_dataloader, val_dataloader, train_num, val_num)
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")

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
        test_dataset = dataset_class(split="test", **dataset_params)
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
