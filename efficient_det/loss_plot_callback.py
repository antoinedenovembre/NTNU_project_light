# ==================================== IMPORTS ==================================== #

# Libraries
import matplotlib.pyplot as plt
import pytorch_lightning as pl

# Custom files
from utils.constants import *

# ===================================== CLASS ===================================== #

class LossPlotCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Log training loss at the end of each epoch
        avg_train_loss = trainer.callback_metrics.get('train_loss', None)
        if avg_train_loss is not None:
            self.train_losses.append(avg_train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log validation loss at the end of each epoch
        avg_val_loss = trainer.callback_metrics.get('val_loss', None)
        if avg_val_loss is not None:
            self.val_losses.append(avg_val_loss.item())

        # Plot the training and validation loss curves after each validation epoch
        plt.figure(figsize=(10, 5))
        if self.train_losses:
            plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

    def on_train_end(self, trainer, pl_module):
        # Plot the training and validation loss curves at the end of training
        plt.figure(figsize=(10, 5))
        if self.train_losses:
            plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        
        # Save the plot
        plt.savefig(LOSS_CURVE_FULL_PATH)