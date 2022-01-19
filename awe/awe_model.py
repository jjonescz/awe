import collections
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import nn
from torch.nn.utils import rnn
from torch_geometric import data
from torchmetrics import functional as metrics

from awe import utils
from awe.data import glove


@dataclass
class AweModelParams:
    label_weights: Sequence[float] = (1,) + (300,) * 4
    use_gnn: bool = False
    use_lstm: bool = False
    use_char_lstm: bool = False
    use_cnn: bool = False
    char_dim: int = 100
    lstm_dim: int = 100
    char_lstm_dim: int = 100
    lstm_args: Optional[dict] = None
    char_lstm_args: Optional[dict] = None
    filter_node_words: bool = True
    label_smoothing: float = 0.0
    pack_words: bool = False
    use_two_gcn_layers: bool = True
    disable_direct_features: bool = False
    use_word_vectors: bool = False

    @property
    def needs_char_embedding(self):
        return self.use_cnn or self.use_char_lstm

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

@dataclass
class ModelInputs:
    batch: data.Batch
    y: Optional[torch.FloatTensor] = None
    z: Optional[torch.FloatTensor] = None

    def compute(self, model: 'AweModel'):
        if self.y is None or self.z is None:
            self.y = self.batch.y
            self.z = model.forward(self.batch)
        return self.y, self.z

# pylint: disable=arguments-differ, unused-argument
class AweModel(pl.LightningModule):
    def __init__(self,
        feature_count: int,
        label_count: int,
        char_count: int,
        params: AweModelParams,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Load pre-trained word embedding layer.
        glove_model = glove.LazyEmbeddings.get_or_create()
        embeddings = torch.FloatTensor(glove_model.vectors)
        # Add one embedding at the top (for unknown and pad words).
        embeddings = F.pad(embeddings, (0, 0, 1, 0))
        self.word_embedding = torch.nn.Embedding.from_pretrained(
            embeddings, padding_idx=0)
        word_dim = embeddings.shape[1]

        if params.needs_char_embedding:
            self.char_embedding = torch.nn.Embedding(
                char_count, params.char_dim)

        if params.use_cnn:
            assert params.use_lstm, 'Cannot use CNN without its parent LSTM'

            self.char_conv = torch.nn.Conv1d(
                in_channels=params.char_dim,
                out_channels=word_dim,
                kernel_size=3,
                padding='same'
            )

            # CNN output will be appended to LSTM input.
            word_dim *= 2

        if params.use_char_lstm:
            assert params.use_lstm, \
                'Cannot use char LSTM without its parent word LSTM'

            self.char_lstm = torch.nn.LSTM(
                params.char_dim,
                params.char_lstm_dim,
                batch_first=True,
                **(params.char_lstm_args or {})
            )

            # Char-LSTM output will be appended to word-LSTM input.
            word_dim += params.char_lstm_dim

        if params.use_lstm:
            self.lstm = torch.nn.LSTM(word_dim, params.lstm_dim,
                batch_first=True,
                **(params.lstm_args or {})
            )

            lstm_out_dim = params.lstm_dim
            if self.lstm.bidirectional:
                # LSTM output will be twice as long.
                lstm_out_dim *= 2

        # Word vector will be appended for each node.
        input_features = feature_count
        if params.use_lstm:
            input_features += lstm_out_dim
        elif params.use_word_vectors:
            input_features += word_dim

        D = 64
        if params.use_gnn:
            self.conv1 = gnn.GCNConv(input_features, D)
            self.conv2 = gnn.GCNConv(D, D)
            # GNN output will be appended to FNN input.
            input_features += D

        self.head = nn.Sequential(
            nn.Linear(input_features, 2 * D),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2 * D, D),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(D, label_count)
        )

        self.loss = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor(params.label_weights),
            label_smoothing=params.label_smoothing
        )

        self.label_count = label_count
        self.params = params

    def forward(self, batch: data.Batch):
        # x: [num_nodes, num_features]
        x, edge_index = batch.x, batch.edge_index

        # Discard feature vector (useful to train only graph structure).
        if self.params.disable_direct_features:
            discarded = torch.ones_like(x)
            # Keep only the first feature (should be relative depth of the node)
            # to keep node vectors at least partly independent.
            discarded[:, 0] = x[:, 0]
            x = discarded

        # Extract character identifiers for the batch.
        char_ids = getattr(batch, 'char_identifiers', None) \
            if self.params.needs_char_embedding else None
            # [num_nodes, max_num_words, max_word_len]
        if char_ids is not None:
            if self.params.filter_node_words:
                masked_char_ids = char_ids[batch.target, :, :]
                    # [num_masked_nodes, max_num_words, max_word_len]
            else:
                masked_char_ids = char_ids

            embedded_chars = self.char_embedding(masked_char_ids)
                # [num_masked_nodes, max_num_words, max_word_len, char_dim]
            num_masked_nodes, max_num_words, max_word_len, char_dim = embedded_chars.shape

            if self.params.use_cnn:
                char_inputs = torch.transpose(embedded_chars, 0, 2)
                    # [max_word_len, max_num_words, num_masked_nodes, char_dim]
                char_inputs = torch.reshape(char_inputs,
                    (max_word_len, -1, char_dim))
                    # [max_word_len, ..., char_dim]
                char_inputs = torch.transpose(char_inputs, 1, 2)
                    # [max_word_len, char_dim, ...]
                char_vectors = self.char_conv(char_inputs)
                    # [max_word_len, word_dim, ...]
                char_vectors = torch.max(char_vectors, dim=0).values
                    # [word_dim, ...]
                char_vectors = torch.transpose(char_vectors, 0, 1)
                    # [..., word_dim]
                char_vectors = torch.reshape(char_vectors,
                    (num_masked_nodes, max_num_words, -1))
                    # [num_masked_nodes, max_num_words, word_dim]
            elif self.params.use_char_lstm:
                char_inputs = torch.reshape(embedded_chars,
                    (-1, max_word_len, char_dim))
                    # [num_words, max_word_len, char_dim]
                _, (char_state, _) = self.char_lstm(char_inputs)
                    # [1, num_words, lstm_char_dim]
                # Keep only the last hidden state (whole word representation).
                char_vectors = char_state[-1, ...] # [num_words, char_lstm_dim]
                char_vectors = torch.reshape(char_vectors,
                    (num_masked_nodes, max_num_words, self.params.char_lstm_dim))

        # Extract word identifiers for the batch.
        word_ids = getattr(batch, 'word_identifiers', None) \
            if self.params.use_word_vectors else None
        if word_ids is not None: # [num_nodes, max_num_words]
            if self.params.filter_node_words:
                # Keep only sentences at leaf nodes.
                masked_word_ids = word_ids[batch.target, :] # [num_masked_nodes, max_num_words]
            else:
                masked_word_ids = word_ids

            # Embed words and pass them through LSTM.
            embedded_words = self.word_embedding(masked_word_ids)
                # [num_masked_nodes, max_num_words, word_dim]

            # Concatenate with character embeddings.
            if self.params.needs_char_embedding:
                embedded_nodes = torch.cat((embedded_words, char_vectors), dim=2)
                    # [num_masked_nodes, max_num_words, 2 * word_dim]
            else:
                embedded_nodes = embedded_words

            if self.params.use_lstm:
                # Pack sequences (to let LSTM ignore pad words).
                if self.params.pack_words:
                    lengths = utils.sequence_lengths(masked_word_ids)
                    if self.params.use_cnn:
                        lengths += char_vectors.shape[-1]
                    packed_nodes = rnn.pack_padded_sequence(
                        embedded_nodes,
                        lengths.cpu(),
                        batch_first = True,
                        enforce_sorted = False
                    )
                else:
                    packed_nodes = embedded_nodes

                # Run through LSTM.
                word_output, (_, _) = self.lstm(packed_nodes)
                    # [num_masked_nodes, max_num_words, D * lstm_dim]

                # Keep only the last word output (whole text representation).
                node_vectors = word_output[:, -1, ...] # [num_masked_nodes, D * lstm_dim]
            else:
                # If not using LSTM, use averaged word embeddings.
                node_vectors = torch.mean(embedded_words, dim=1) # [num_masked_nodes, word_dim]

            if self.params.filter_node_words:
                # Expand word vectors to the original shape.
                full_node_vectors = torch.zeros(word_ids.shape[0], node_vectors.shape[1])
                    # [num_nodes, lstm_dim]
                full_node_vectors = full_node_vectors.type_as(x)
                full_node_vectors[batch.target] = node_vectors
            else:
                full_node_vectors = node_vectors

            # Append to features.
            x = torch.hstack((x, full_node_vectors)) # [num_nodes, num_features]

        # Propagate features through edges (graph convolution).
        if self.params.use_gnn:
            orig = x # [num_nodes, num_features]
            x = self.conv1(x, edge_index) # [num_nodes, D]
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            if self.params.use_two_gcn_layers:
                x = self.conv2(x, edge_index)

        # Filter target nodes (we want to propagate features through all edges
        # but classify only leaf nodes).
        if self.params.use_gnn:
            orig = orig[batch.target] # [num_target_nodes, num_features]
        x = x[batch.target] # [num_target_nodes, D]

        # Concatenate original feature vector with convoluted feature vector.
        if self.params.use_gnn:
            x = torch.hstack((orig, x)) # [num_target_nodes, num_features + D]

        # Classify using deep fully connected head.
        x = self.head(x) # [num_target_nodes, num_classes]

        # Un-filter target nodes (so dimensions are as expected).
        full = torch.zeros(batch.x.shape[0], x.shape[1]) # [num_nodes, num_classes]
        full = full.type_as(x)
        full[:, 0] = 1 # classify as "none" by default (for non-target nodes)
        full[batch.target] = x # use computed classification of target nodes
        return full

    def training_step(self, batch: data.Batch, *_):
        y = batch.y
        z = self.forward(batch)
        loss = self.criterion(z, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: data.Batch, _, idx: int = 0):
        return self._shared_eval_step('val', batch, idx)

    def test_step(self, batch: data.Batch, _, idx: int = 0):
        return self._shared_eval_step('test', batch, idx)

    def _shared_eval_step(self, prefix: str, batch: data.Batch, idx: int):
        y = batch.y
        z = self.forward(batch)
        loss = self.criterion(z, y)
        preds = torch.argmax(z, dim=1)

        acc = metrics.accuracy(preds, y)
        f1 = metrics.f1_score(preds, y,
            average="weighted",
            num_classes=self.label_count,
            ignore_index=0
        )
        swde_f1 = self.compute_swde_f1(ModelInputs(batch, y, z))

        results = {
            'loss': loss,
            'acc': acc,
            'f1': f1,
            'swde_f1': swde_f1
        }
        prefixed = { f'{prefix}_{key}': value for key, value in results.items() }
        log_args = {
            'prog_bar': (idx == 0),
            'add_dataloader_idx': (idx != 0)
        }
        self.log_dict(prefixed, **log_args)

        # Log `hp_metric` which is used as main metric in TensorBoard.
        if prefix == 'val':
            self.log('hp_metric', swde_f1, **(log_args | {'prog_bar': False}))

        return prefixed

    def predict_step(self, batch: data.Batch, *_):
        z = self.forward(batch)
        preds = torch.argmax(z, dim=1)
        return preds

    def criterion(self, z, y):
        return self.loss(z, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def compute_swde_f1(self, inputs: ModelInputs):
        swde_metrics = [
            self.compute_swde_metrics(inputs, label)
            for label in range(self.label_count)
            if label != 0
        ]
        swde_f1s = [m.f1 for m in swde_metrics]
        return np.mean(swde_f1s)

    def predict_swde(self, inputs: ModelInputs, label: int, callback: Callable):
        """SWDE-inspired prediction computation: per-attribute, page-wide."""

        batch = inputs.batch
        y, z = inputs.compute(self)
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

    def compute_swde_metrics(self, inputs: ModelInputs, label: int):
        """SWDE-inspired metric computation: per-attribute, page-wide."""

        stats = collections.defaultdict(int)
        def increment(name: str, *_):
            stats[name] += 1
        self.predict_swde(inputs, label, increment)

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

    def get_progress_bar_dict(self):
        # Parse version number properly.
        version = self.trainer.logger.version
        if isinstance(version, str):
            version = version.split('-')[0]

        items = super().get_progress_bar_dict()
        items['v_num'] = version
        return items
