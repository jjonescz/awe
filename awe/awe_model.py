import pytorch_lightning as pl
from torchmetrics import functional as metrics
import torch
import torch.nn.functional as F
from torch import nn


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

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.argmax(y, dim=1)
        z = self.forward(x)
        loss = self.criterion(z, y)
        # self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.argmax(y, dim=1)
        z = self.forward(x)
        loss =  self.criterion(z, y)
        preds = torch.argmax(z, dim=1)
        acc = metrics.accuracy(preds, y)
        nt_acc = 0 if sum(y != 0) == 0 else metrics.accuracy(preds[y != 0], y[y != 0])
        f1 = metrics.f1(preds, y, average="weighted", num_classes=self.label_count, ignore_index=0)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_nt_acc", nt_acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        return loss

    def criterion(self, z, y):
        return F.cross_entropy(z, y, weight=self.label_weights)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
