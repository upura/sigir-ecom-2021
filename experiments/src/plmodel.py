import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from typing import List

from src.models.transformer_encoder import TransformerEncoderModel
from src.metrics import evaluate_rec_task_metrics


class RecTaskPLModel(pl.LightningModule):
    def __init__(self, config: dict, num_labels: int) -> None:
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.model = TransformerEncoderModel(
            encoder_params=config["encoder_params"],
            num_labels=num_labels,
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x_batch, device: torch.device):
        output = self.model(**x_batch.to_dict(device))
        return output

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_pred = self.forward(x_batch, self.config["device"])

        loss = self.criterion(y_pred, y_batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_pred = self.forward(x_batch, self.config["device"])

        loss = self.criterion(y_pred, y_batch)
        return {"loss": loss}

    def training_epoch_end(self, train_step_outputs: List[dict]):
        loss = torch.stack([o["loss"] for o in train_step_outputs]).mean()

        self.log("step", self.current_epoch)
        self.log("train_loss", loss)

    def validation_epoch_end(self, val_step_outputs: List[dict]):
        val_loss = torch.stack([o["loss"] for o in val_step_outputs]).mean()

        self.log("step", self.current_epoch)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            **self.config["optimizer_params"],
        )
        return optimizer
