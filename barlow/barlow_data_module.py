# ==================================== IMPORTS ==================================== #

# Libraries
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch

# Custom files
from barlow.barlow_dataset import BarlowTwinsDataset
from utils.functions import *
from utils.constants import *
from utils.logger import _app_logger

# ===================================== CLASS ===================================== #

class BarlowTwinsDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset_adaptor,
        validation_dataset_adaptor,
        train_transforms,
        valid_transforms,
        num_workers=4,
        batch_size=8,
    ):
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> BarlowTwinsDataset:
        return BarlowTwinsDataset(
            dataset_adaptor=self.train_ds, transforms=self.train_tfms
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataset(self) -> BarlowTwinsDataset:
        return BarlowTwinsDataset(
            dataset_adaptor=self.valid_ds, transforms=self.valid_tfms
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return valid_loader

    @staticmethod
    def collate_fn(batch):
        x1, x2, x3, image_ids = tuple(zip(*batch))
        x1, x2, x3 = (
            torch.stack(x1).float(),
            torch.stack(x2).float(),
            torch.stack(x3).float(),
        )

        return (x1, x2, x3), torch.tensor(image_ids)