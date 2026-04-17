import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class ParquetDataset(Dataset):
    """
    Dataset that loads data from a Parquet file.
    Suitable for PINNs or supervised learning.
    """

    def __init__(self, parquet_path:Path, input_cols:list[str], target_cols:list[str]):
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
    Wrapper that builds a PyTorch DataLoader from a Parquet dataset.
    """

    def __init__(self,
                 parquet_path,
                 input_cols,
                 target_cols,
                 batch_size=32,
                 shuffle=True,
                 num_workers=0,
                 pin_memory=True):

        self.dataset = ParquetDataset(
            parquet_path=parquet_path,
            input_cols=input_cols,
            target_cols=target_cols
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def get_loader(self):
        return self.dataloader