# ==================================== IMPORTS ==================================== #

# Libraries
from pytorch_lightning import LightningModule
import torch

# Custom files
from barlow.barlow_loss import BarlowTwinsLoss
from barlow.projection_head import ProjectionHead
from utils.functions import *
from utils.constants import *
from utils.logger import _app_logger

# ===================================== CLASS ===================================== #

class BarlowTwins(LightningModule):
    def __init__(
        self,
        encoder,
        encoder_out_dim,
        get_repr,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        z_dim=128,
        learning_rate=1e-4,
        warmup_epochs=10,
        max_epochs=200,
    ):
        super().__init__()

        self.encoder = encoder
        self.get_repr = get_repr
        self.projection_head = ProjectionHead(
            input_dim=encoder_out_dim, output_dim=z_dim
        )
        self.loss_fn = BarlowTwinsLoss(
            batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim
        )

        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.train_iters_per_epoch = num_training_samples // batch_size

    def forward(self, x):
        x = self.encoder(x)[0]
        x = self.get_repr(x)
        return x

    def shared_step(self, batch):
        (x1, x2, _), _ = batch

        x1 = self.encoder(x1)[0]
        x2 = self.encoder(x2)[0]

        x1 = self.get_repr(x1).squeeze()
        x2 = self.get_repr(x2).squeeze()

        z1 = self.projection_head(x1)
        z2 = self.projection_head(x2)

        return self.loss_fn(z1, z2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]