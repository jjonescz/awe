import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric import data
from torchmetrics import functional as metrics


class AweModel(pl.LightningModule):
    def __init__(self, label_count, label_weights):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
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
        y = batch.y
        z = self.forward(batch.x)
        loss =  self.criterion(z, y)
        preds_conf, preds = torch.max(z, dim=1)

        # SWDE-inspired metric computation: per-attribute, page-wide.
        def compute_swde_metrics(label):
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            for page in range(batch.num_graphs):
                # Filter for the page and label.
                mask = torch.logical_and(batch.batch == page, preds == label)
                curr_preds_conf = preds_conf[mask]

                if len(curr_preds_conf) == 0:
                    if (y[batch.batch == page] == label).sum() == 0:
                        true_negatives += 1
                    else:
                        false_negatives += 1
                    continue

                # Find only the most confident prediction.
                idx = torch.argmax(curr_preds_conf, dim=0)
                curr_preds_conf = curr_preds_conf[idx]

                # Is the attribute correctly extracted?
                if y[mask][idx] == label:
                    true_positives += 1
                else:
                    false_positives += 1

            if (true_positives + false_positives) == 0:
                precision = 0
            else:
                precision = true_positives / (true_positives + false_positives)
            if (true_positives + false_negatives) == 0:
                recall = 0
            else:
                recall = true_positives / (true_positives + false_negatives)
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            return precision, recall, f1

        swde_metrics = [
            compute_swde_metrics(label)
            for label in range(self.label_count)
        ]
        swde_f1s = [m[2] for m in swde_metrics]
        swde_f1 = np.mean(swde_f1s)

        acc = metrics.accuracy(preds, y)
        f1 = metrics.f1(preds, y, average="weighted", num_classes=self.label_count, ignore_index=0)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_swde_f1", swde_f1, prog_bar=True)
        return loss

    def criterion(self, z, y):
        return F.cross_entropy(z, y, weight=self.label_weights)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
