# ==================================== IMPORTS ==================================== #

# Libraries
from torch.utils.data import Dataset
import torch

# Custom files
from utils.functions import *
from utils.constants import *
from utils.logger import _app_logger

# ===================================== CLASS ===================================== #

class BarlowTwinsDataset(Dataset):
    def __init__(self, dataset_adaptor, transforms):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        image = self.ds.get_image_by_idx(index)
        return *self.transforms(image), index

    def __len__(self):
        return len(self.ds)