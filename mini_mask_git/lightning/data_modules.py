import json
import numpy as np
import torch
import pytorch_lightning as pl

from pathlib import Path
from typing import Optional, Union, List, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split, DataLoader, Dataset


from mini_mask_git.data import LatentImageDataset


class LitLatentImageDatamodule(pl.LightningDataModule):
    def __init__(self,
                 root: str,
                 num_embeds: int,
                 spatial_size: Tuple[int, int],
                 train_file: str = 'train',
                 val_file: str = 'val',
                 test_file: str = 'test',
                 counts_file: str = 'counts.npy',
                 batch_size: int = 32,
                 num_workers: int = 8,
                 dummy: bool = False):
        super().__init__()

        self.save_hyperparameters()

        self.root = Path(root)
        self.num_embeds = num_embeds
        self.spatial_size = spatial_size
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.counts_file = counts_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dummy = dummy

        self.counts = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_dataset(self, path, mask=False):
        return LatentImageDataset(
            path,
            mask=mask,
            dummy=self.dummy
        )

    def setup(self, stage: Optional[str] = None):
        self.counts = torch.tensor(np.load(str(self.root / self.counts_file)))
        self.counts /= self.counts.sum()

        self.train_data = self.load_dataset(self.root / self.train_file, mask=True)
        self.val_data = self.load_dataset(self.root / self.val_file, mask=True)
        self.test_data = self.load_dataset(self.root / self.test_file, mask=True)

    def create_dataloader(self, dataset, shuffle=False, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last
        )

    def train_dataloader(self):
        return self.create_dataloader(self.train_data, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self.create_dataloader(self.val_data, shuffle=True)

    def test_dataloader(self):
        return self.create_dataloader(self.test_data, shuffle=True)
