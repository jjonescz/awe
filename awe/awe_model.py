import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class AweModel(pl.LightningModule):
    def __init__(self, label_count):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, label_count)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.forward(x)
        loss = F.cross_entropy(z, torch.max(y, 1)[1])
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
