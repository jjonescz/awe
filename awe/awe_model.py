import collections
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import nn
from torch_geometric import data
from torchmetrics import functional as metrics


@dataclass
class SwdeMetrics:
    precision: float
    recall: float
    f1: float

    def to_vector(self):
        return torch.FloatTensor([self.precision, self.recall, self.f1])

    @staticmethod
    def from_vector(vector: torch.FloatTensor):
        return SwdeMetrics(vector[0].item(), vector[1].item(), vector[2].item())

# pylint: disable=arguments-differ, unused-argument
class AweModel(pl.LightningModule):
    def __init__(self,
        feature_count: int,
        label_count: int,
        label_weights: list[float],
        char_count: int,
        use_gnn: bool,
        char_dim: int = 100
    ):
        super().__init__()

        self.save_hyperparameters()

        self.char_embedding = torch.nn.Embedding(char_count, char_dim)
        self.char_conv = torch.nn.Conv1d(
            in_channels=char_dim,
            out_channels=50,
            kernel_size=3,
            padding='same'
        )

        D = 64
        if use_gnn:
            self.conv1 = gnn.GCNConv(feature_count, D)
            self.conv2 = gnn.GCNConv(D, D)
        input_dim = feature_count + D if use_gnn else feature_count
        self.head = nn.Sequential(
            nn.Linear(input_dim, 2 * D),
            nn.ReLU(),
            nn.Linear(2 * D, D),
            nn.ReLU(),
            nn.Linear(D, label_count)
        )

        self.label_count = label_count
        self.label_weights = torch.FloatTensor(label_weights)
        self.use_gnn = use_gnn

    def forward(self, batch: data.Data):
        # x: [num_nodes, num_features]
        x, edge_index = batch.x, batch.edge_index

        # Propagate features through edges (graph convolution).
        if self.use_gnn:
            x = self.conv1(x, edge_index) # [num_nodes, D]
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)

        # Filter target nodes (we want to propagate features through all edges
        # but classify only leaf nodes).
        if self.use_gnn:
            orig = batch.x[batch.target] # [num_target_nodes, num_features]
        x = x[batch.target] # [num_target_nodes, D]

        # Concatenate original feature vector with convoluted feature vector.
        if self.use_gnn:
            x = torch.hstack((orig, x)) # [num_target_nodes, D + num_features]

        # Classify using deep fully connected head.
        x = self.head(x) # [num_target_nodes, num_classes]

        # Un-filter target nodes (so dimensions are as expected).
        full = torch.zeros(batch.x.shape[0], x.shape[1]) # [num_nodes, num_classes]
        full[:, 0] = 1 # classify as "none" by default (for non-target nodes)
        full[batch.target] = x # use computed classification of target nodes
        return full

    def training_step(self, batch: data.Batch, batch_idx: int):
        y = batch.y
        z = self.forward(batch)
        loss = self.criterion(z, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: data.Batch, batch_idx: int):
        return self._shared_eval_step('val', batch, batch_idx)

    def test_step(self, batch: data.Batch, batch_idx: int):
        return self._shared_eval_step('test', batch, batch_idx)

    def _shared_eval_step(self, prefix: str, batch: data.Batch, batch_idx: int):
        swde_metrics = [
            self.compute_swde_metrics(batch, label)
            for label in range(self.label_count)
            if label != 0
        ]
        swde_f1s = [m.f1 for m in swde_metrics]
        swde_f1 = np.mean(swde_f1s)

        y = batch.y
        z = self.forward(batch)
        loss = self.criterion(z, y)
        preds = torch.argmax(z, dim=1)

        acc = metrics.accuracy(preds, y)
        f1 = metrics.f1(preds, y, average="weighted", num_classes=self.label_count, ignore_index=0)

        results = {
            'loss': loss,
            'acc': acc,
            'f1': f1,
            'swde_f1': swde_f1
        }
        prefixed = { f'{prefix}_{key}': value for key, value in results.items() }
        self.log_dict(prefixed, prog_bar=True)
        return prefixed

    def predict_step(self, batch: data.Batch, batch_idx: int):
        z = self.forward(batch)
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
        z = self.forward(batch)
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
        return SwdeMetrics(precision, recall, f1)
