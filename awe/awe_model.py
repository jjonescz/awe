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
        preds = torch.argmax(z, dim=1)

        def metric(f, mask):
            return f(preds[mask], y[mask])

        def page_metrics(f):
            """Computes metrics for each page (graph) separately."""
            return [
                metric(f, batch.batch == page)
                for page in range(batch.num_graphs)
            ]

        accs = page_metrics(metrics.accuracy)
        f1s = page_metrics(metrics.f1)
        acc = metrics.accuracy(preds, y)
        f1 = metrics.f1(preds, y, average="weighted", num_classes=self.label_count, ignore_index=0)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_page_acc", np.mean(accs), prog_bar=True)
        self.log("val_page_f1", np.mean(f1s), prog_bar=True)
        return loss

    def criterion(self, z, y):
        return F.cross_entropy(z, y, weight=self.label_weights)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
