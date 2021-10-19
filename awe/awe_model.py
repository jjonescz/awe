import collections
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric import data
from torchmetrics import functional as metrics


class AweModel(pl.LightningModule):
    def __init__(self, feature_count, label_count, label_weights):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_count, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, label_count)
        )
        self.label_count = label_count
        self.label_weights = torch.FloatTensor(label_weights)

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    def training_step(self, batch: data.Batch, batch_idx: int):
        y = batch.y
        z = self.forward(batch.x)
        loss = self.criterion(z, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: data.Batch, batch_idx: int):
        swde_metrics = [
            self.compute_swde_metrics(batch, label)
            for label in range(self.label_count)
            if label != 0
        ]
        swde_f1s = [m[2] for m in swde_metrics]
        swde_f1 = np.mean(swde_f1s)

        y = batch.y
        z = self.forward(batch.x)
        loss = self.criterion(z, y)
        preds = torch.argmax(z, dim=1)

        acc = metrics.accuracy(preds, y)
        f1 = metrics.f1(preds, y, average="weighted", num_classes=self.label_count, ignore_index=0)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_swde_f1", swde_f1, prog_bar=True)
        return loss

    def predict_step(self, batch: data.Batch, batch_idx: int):
        z = self.forward(batch.x)
        preds = torch.argmax(z, dim=1)
        return preds

    def criterion(self, z, y):
        return F.cross_entropy(z, y, weight=self.label_weights)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_swde(self, batch: data.Batch, label: int, callback: Callable):
        """SWDE-inspired prediction computation: per-attribute, page-wide."""

        y = batch.y
        z = self.forward(batch.x)
        preds_conf, preds = torch.max(z, dim=1)

        for page in range(batch.num_graphs):
            # Filter for the page and label.
            mask = torch.logical_and(batch.batch == page, preds == label)
            curr_preds_conf = preds_conf[mask]

            if len(curr_preds_conf) == 0:
                if (y[batch.batch == page] == label).sum() == 0:
                    callback('tn', mask)
                else:
                    callback('fn', mask)
                continue

            # Find only the most confident prediction.
            idx = torch.argmax(curr_preds_conf, dim=0)
            curr_preds_conf = curr_preds_conf[idx]

            # Is the attribute correctly extracted?
            if y[mask][idx] == label:
                callback('tp', mask, idx)
            else:
                callback('fp', mask, idx)

    def compute_swde_metrics(self, batch: data.Batch, label: int):
        """SWDE-inspired metric computation: per-attribute, page-wide."""

        stats = collections.defaultdict(int)
        def increment(name: str, *_):
            stats[name] += 1
        self.predict_swde(batch, label, increment)

        true_positives = stats['tp']
        false_positives = stats['fp']
        false_negatives = stats['fn']
        if (true_positives + false_positives) == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)
        if (true_positives + false_negatives) == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1