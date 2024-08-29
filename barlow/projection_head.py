# ==================================== IMPORTS ==================================== #

# Libraries
import torch.nn as nn

# Custom files
from utils.functions import *
from utils.constants import *
from utils.logger import _app_logger

# ===================================== CLASS ===================================== #

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=8192):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)