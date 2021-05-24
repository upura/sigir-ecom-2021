import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from typing import List

from src.models.transformer_encoder import TransformerEncoderModel
from src.metrics import evaluate_rec_task_metrics
from src.loss import FocalLoss


class RecTaskPLModel(pl.LightningModule):
    def __init__(self, config: dict, num_labels: int) -> None:
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.model = TransformerEncoderModel(
            encoder_params=config["encoder_params"],
            num_labels=num_labels,
        )
        self.criterion = FocalLoss(
            gamma=config["loss_func_params"]["gamma"],
            alpha=config["loss_func_params"]["alpha"],
        )

    def forward(self, x_batch, device: torch.device):
        output_next_item, output_subsequent_items = self.model(**x_batch.to_dict(device))
        return output_next_item, output_subsequent_items

    def training_step(self, batch, batch_idx):
        x_batch, y_batch_next_item, y_batch_subsequent_items = batch
        y_pred_next_item, y_pred_subsequent_items = self.forward(x_batch, self.config["device"])

        loss_next_item = self.criterion(y_pred_next_item, y_batch_next_item)
        loss_subsequent_items = self.criterion(y_pred_subsequent_items, y_batch_subsequent_items)
        loss = loss_next_item + loss_subsequent_items
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch_next_item, y_batch_subsequent_items = batch
        y_pred_next_item, y_pred_subsequent_items = self.forward(x_batch, self.config["device"])

        loss_next_item = self.criterion(y_pred_next_item, y_batch_next_item)
        loss_subsequent_items = self.criterion(y_pred_subsequent_items, y_batch_subsequent_items)
        loss = loss_next_item + loss_subsequent_items

        y_pred = y_pred_next_item + y_pred_subsequent_items
        metrics = evaluate_rec_task_metrics(
            y_pred,
            y_batch_next_item,
            y_batch_subsequent_items,
        )
        metrics["loss"] = loss
        return metrics

    def training_epoch_end(self, train_step_outputs: List[dict]):
        loss = torch.stack([o["loss"] for o in train_step_outputs]).mean()

        self.log("step", self.current_epoch)
        self.log("train_loss", loss)

    def validation_epoch_end(self, val_step_outputs: List[dict]):
        val_loss = torch.stack([o["loss"] for o in val_step_outputs]).mean()
        val_f1_score = np.mean([o["f1_score"] for o in val_step_outputs])
        val_mrr = np.mean([o["mrr"] for o in val_step_outputs])

        self.log("step", self.current_epoch)
        self.log("val_loss", val_loss)
        self.log("val_f1_score", val_f1_score)
        self.log("val_mrr", val_mrr)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            **self.config["optimizer_params"],
        )
        return optimizer
