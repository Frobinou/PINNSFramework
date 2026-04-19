import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


class ParquetDataset(Dataset):
    """
    Dataset that loads data from a Parquet file.
    Suitable for PINNs or supervised learning.
    """

    def __init__(self, parquet_path: Path, input_cols: list[str], target_cols: list[str]):
        """
        Args:
            parquet_path (str): path to .parquet file
            input_cols (list): feature columns (x)
            target_cols (list): target columns (y)
        """

        self.df = pd.read_parquet(parquet_path)
        self.input_cols = input_cols
        self.target_cols = target_cols

        # Convert once to numpy for efficiency
        self.x = self.df[input_cols].values.astype("float32")
        self.y = self.df[target_cols].values.astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx]),
            "y": torch.tensor(self.y[idx]),
        }


class ParquetDataLoader:
    """
    Wrapper that builds train/val/test DataLoaders from a Parquet dataset.
    """

    def __init__(
        self,
        parquet_path,
        input_cols,
        target_cols,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        train_ratio=0.7,
        val_ratio=0.15,
        # test_ratio is implicit: 1 - train_ratio - val_ratio
        seed=42,
    ):

        assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be < 1"

        dataset = ParquetDataset(
            parquet_path=parquet_path,
            input_cols=input_cols,
            target_cols=target_cols,
        )

        n = len(dataset)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds, test_ds = random_split(
            dataset, [n_train, n_val, n_test], generator=generator
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader
