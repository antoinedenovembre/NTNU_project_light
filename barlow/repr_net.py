# ==================================== IMPORTS ==================================== #

# Libraries
from typing import List
import torch.nn as nn
import torch

# Custom files
from utils.functions import *
from utils.constants import *
from utils.logger import _app_logger

# ===================================== CLASS ===================================== #

class ReprNet(nn.Module):
    def __init__(self, config):
        super(ReprNet, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(config.num_levels):
            self.conv.append(
                nn.AvgPool2d(2 ** (config.max_level - i - 1)),
            )

    def forward(self, x: List[torch.Tensor]):
        outputs = []
        for level, x_l in enumerate(x):
            outputs.append(self.conv[level](x_l))

        return sum(outputs)