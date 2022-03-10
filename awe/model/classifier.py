import dataclasses
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn

import awe.data.glove
import awe.data.graph.dom
import awe.features.dom
import awe.features.extraction
import awe.model.word_lstm

if TYPE_CHECKING:
    import awe.training.trainer

ModelInput = list[awe.data.graph.dom.Node]

@dataclasses.dataclass
class ModelOutput:
    loss: torch.FloatTensor
    logits: torch.FloatTensor
    gold_labels: torch.FloatTensor

    def get_pred_labels(self):
        return torch.argmax(self.logits, dim=-1)

@dataclasses.dataclass
class Prediction:
    batch: ModelInput
    outputs: ModelOutput

class Model(torch.nn.Module):
    def __init__(self,
        trainer: 'awe.training.trainer.Trainer',
        lr: Optional[float] = None,
    ):
        super().__init__()
        self.trainer = trainer
        self.lr = lr

        node_features = 0

        # HTML tag name embedding
        self.html_tag = self.trainer.extractor.get_feature(awe.features.dom.HtmlTag)
        if self.html_tag is not None:
            num_html_tags = len(self.html_tag.html_tags) + 1
            embedding_dim = 32
            self.tag_embedding = torch.nn.Embedding(
                num_html_tags,
                embedding_dim
            )
            node_features += embedding_dim

        self.position = self.trainer.extractor.get_feature(awe.features.dom.Position)
        if self.position is not None:
            node_features += 2

        # Word LSTM
        if self.trainer.params.word_vector_function is not None:
            self.lstm = awe.model.word_lstm.WordLstm(self)
            out_dim = self.lstm.out_dim
            if self.trainer.params.friend_cycles:
                out_dim *= 3
            node_features += out_dim

        # Visual neighbors
        if self.trainer.params.visual_neighbors:
            self.neighbor_attention = torch.nn.Linear(node_features * 2 + 3, 1)
            head_features = node_features * 2
        else:
            head_features = node_features

        # Classification head
        D = 64
        num_labels = len(self.trainer.label_map.id_to_label) + 1
        self.head = torch.nn.Sequential(
            torch.nn.Linear(head_features, 2 * D),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(2 * D, D),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(D, num_labels)
        )

        self.loss = torch.nn.CrossEntropyLoss(
            #weight=torch.FloatTensor(params.label_weights),
            #label_smoothing=params.label_smoothing
        )

    def create_optimizer(self):
        return torch.optim.Adam(self.parameters(),
            lr=(self.lr or self.trainer.params.learning_rate)
        )

    def get_node_features_slim(self, batch: list[awe.data.graph.dom.Node]):
        """
        Features used for a node, its ancestor chain, and its visual neighbors.
        """
        x = None

        # Embed HTML tag names.
        if self.html_tag is not None:
            tag_ids = self.html_tag.compute(batch) # [N]
            x = append(x, self.tag_embedding(tag_ids)) # [N, embedding_dim]

        return x

    def get_node_features(self, batch: list[awe.data.graph.dom.Node]):
        """Features used for a node and its visual neighbors."""

        x = self.get_node_features_slim(batch)

        # Add more HTML node features.
        if self.position is not None:
            x = append(x, self.position.compute(batch))

        if self.trainer.params.word_vector_function is not None:
            if self.trainer.params.friend_cycles:
                # Expand partner and friend nodes.
                friend_batch = [None] * (len(batch) * 3)
                for i, n in zip(range(0, len(friend_batch), 3), batch):
                    friend_batch[i] = [n]
                    friend_batch[i + 1] = n.get_partner_set()
                    friend_batch[i + 2] = n.friends
                expanded_batch = friend_batch
            else:
                expanded_batch = [[n] for n in batch]

            y = self.lstm(expanded_batch) # [3N, lstm_dim]

            if self.trainer.params.friend_cycles:
                y = torch.reshape(y, (len(batch), y.shape[1] * 3)) # [N, 3lstm_dim]

            x = append(x, y) # [N, node_features]

        return x

    def forward(self, batch: ModelInput) -> ModelOutput:
        # Propagate visual neighbors.
        if self.trainer.params.visual_neighbors:
            n_neighbors = self.trainer.params.n_neighbors
            neighbors = [
                v
                for n in batch
                for v in n.visual_neighbors
            ]
            neighbor_batch = [v.neighbor for v in neighbors]

            node_features = self.get_node_features(batch) # [N, node_features]
            neighbor_features = self.get_node_features(neighbor_batch)
                # [n_neighbors * N, node_features]
            distances = torch.tensor(
                [
                    (v.distance_x, v.distance_y, v.distance)
                    for v in neighbors
                ],
                dtype=torch.float32,
                device=self.trainer.device
            ) # [n_neighbors * N, 3]

            # Get node for each neighbor (e.g., [0, 0, 1, 1, 2, 2] if there are
            # three nodes and each has two neighbors)
            expanded_features = node_features.repeat_interleave(n_neighbors, dim=0)
                # [n_neighbors * N, node_features]

            # Compute neighbor coefficients (for each node, its neighbor, and
            # distance between them).
            coefficient_inputs = torch.concat(
                (expanded_features, neighbor_features, distances),
                dim=-1
            ) # [n_neighbors * N, 2*node_features + 3]
            coefficients = self.neighbor_attention(coefficient_inputs)
                # [n_neighbors * N, 1]

            # Aggregate neighbor features (sum weighted by the coefficients).
            coefficients = coefficients.reshape((len(batch), 1, n_neighbors))
                # [N, 1, n_neighbors]
            neighbor_features = neighbor_features \
                .reshape((len(batch), n_neighbors, -1))
                # [N, n_neighbors, node_features]
            neighborhood = torch.matmul(coefficients, neighbor_features)
                # [N, 1, node_features]
            neighborhood = neighborhood.reshape((len(batch), -1))
                # [N, node_features]

            x = torch.concat((node_features, neighborhood), dim=-1)
                # [N, 2 * node_features]
        else:
            x = self.get_node_features(batch)

        # Classify features.
        x = self.head(x) # [N, num_labels]

        # Find out gold labels.
        gold_labels = torch.tensor(
            [
                self.trainer.label_map.get_label_id(node)
                for node in batch
            ],
            device=self.trainer.device
        ) # [num_labels]
        loss = self.loss(x, gold_labels)

        return ModelOutput(
            loss=loss,
            logits=x,
            gold_labels=gold_labels
        )

def append(x: Optional[torch.Tensor], y: torch.Tensor):
    if x is None:
        return y
    return torch.concat((x, y), dim=-1)
