import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

from src.core.registry import REGISTRY
from src.repositories.data_loader.base_dataloader import BaseDataLoader

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


@REGISTRY.data_loaders.register("parquet")  
class ParquetDataLoader(BaseDataLoader):

    def __init__(self, data_path: Path, input_cols: list[str], target_cols: list[str], **kwargs):
        self.parquet_path = Path(data_path)
        self.input_cols = input_cols
        self.target_cols = target_cols
        super().__init__(**kwargs)  # déclenche build_dataset() + split

    def build_dataset(self) -> Dataset:
        return ParquetDataset(
            parquet_path=self.parquet_path,
            input_cols=self.input_cols,
            target_cols=self.target_cols,
        )
