import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as plf
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
        self.label_weights = torch.FloatTensor(label_weights)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.forward(x)
        loss = self.criterion(z, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.forward(x)
        loss =  self.criterion(z, y)
        preds = torch.argmax(z, dim=1)
        acc = plf.accuracy(preds, torch.max(y, 1)[1])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def criterion(self, z, y):
        return F.cross_entropy(z, torch.max(y, 1)[1], weight=self.label_weights)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
