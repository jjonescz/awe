import dataclasses
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn
import torch.nn.functional as F

import awe.data.glove
import awe.data.graph.dom
import awe.features.dom
import awe.features.extraction
import awe.features.text
import awe.features.visual
import awe.model.word_lstm
import awe.training.params

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

    def get_probabilities(self):
        return F.softmax(self.logits, dim=-1)

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

        self.slim_node_feature_dim = 0

        self.dropout = torch.nn.Dropout(0.3)

        # Word embedding (shared for node text and attributes)
        self.word_ids = self.trainer.extractor.get_feature(awe.features.text.WordIdentifiers)
        if self.word_ids is not None:
            self.word_embedding = awe.model.word_lstm.WordEmbedding(self)

        # HTML tag name embedding
        self.html_tag = self.trainer.extractor.get_feature(awe.features.dom.HtmlTag)
        if self.html_tag is not None:
            num_html_tags = len(self.html_tag.html_tag_ids) + 1
            embedding_dim = self.trainer.params.tag_name_embedding_dim
            self.tag_embedding = torch.nn.Embedding(
                num_html_tags,
                embedding_dim
            )
            self.slim_node_feature_dim += embedding_dim

        self.node_feature_dim = self.slim_node_feature_dim

        if self.trainer.params.tokenize_node_attrs:
            self.slim_node_feature_dim += self.word_embedding.out_dim

            if not self.trainer.params.tokenize_node_attrs_only_ancestors:
                self.node_feature_dim += self.word_embedding.out_dim

        # Node position
        self.position = self.trainer.extractor.get_feature(awe.features.dom.Position)
        if self.position is not None:
            self.node_feature_dim += self.position.out_dim

        # Node visuals
        self.visuals = self.trainer.extractor.get_feature(awe.features.visual.Visuals)
        if self.visuals is not None:
            self.node_feature_dim += self.visuals.out_dim

        # Word LSTM
        if self.trainer.params.word_vector_function is not None:
            self.lstm = awe.model.word_lstm.WordLstm(self)
            out_dim = self.lstm.out_dim
            if self.trainer.params.friend_cycles:
                out_dim *= 3
            self.node_feature_dim += out_dim

        # Visual neighbors
        if self.trainer.params.visual_neighbors:
            self.neighbor_attention = torch.nn.Linear(
                self.node_feature_dim * 2 + 3, 1
            )
            self.neighbor_softmax = torch.nn.Softmax(dim=-1)
            head_features = self.node_feature_dim * 2
        else:
            head_features = self.node_feature_dim

        # Ancestor chain
        if self.trainer.params.ancestor_chain:
            # For each ancestor we have slim features and its distance (the
            # latter only except LSTM, it learns distance differently).
            self.ancestor_dist = self.trainer.params.ancestor_function == 'lstm'
            input_ancestor_dim = self.slim_node_feature_dim + (
                1 if self.ancestor_dist else 0
            )

            # We can summarize features of each ancestor using one layer.
            if self.trainer.params.ancestor_summarize:
                self.ancestor_dim = self.trainer.params.ancestor_dim or input_ancestor_dim
                self.ancestor_layer = torch.nn.Linear(input_ancestor_dim, self.ancestor_dim)
            else:
                self.ancestor_dim = input_ancestor_dim

            # We then aggregate features for each group of ancestors.
            if self.trainer.params.ancestor_function == 'lstm':
                out_dim = (self.trainer.params.ancestor_lstm_out_dim or
                    self.ancestor_dim)
                self.ancestor_lstm = torch.nn.LSTM(
                    self.ancestor_dim,
                    out_dim,
                    **(self.trainer.params.ancestor_lstm_args or {})
                )
                if self.ancestor_lstm.bidirectional:
                    out_dim *= 2
                head_features += out_dim
            elif self.trainer.params.ancestor_function == 'attention':
                self.ancestor_attention = torch.nn.Linear(self.ancestor_dim, 1)
                self.ancestor_softmax = torch.nn.Softmax(dim=-1)
                head_features += self.ancestor_dim
            else:
                head_features += self.ancestor_dim

        # Classification head
        num_labels = len(self.trainer.label_map.id_to_label) + 1
        head_layers = []
        head_dims = [head_features]
        head_dims.extend(self.trainer.params.head_dims)
        for prev_dim, next_dim in zip(head_dims, head_dims[1:]):
            head_layers.append(torch.nn.Linear(prev_dim, next_dim))
            if self.trainer.params.layer_norm:
                head_layers.append(torch.nn.LayerNorm(next_dim))
            head_layers.append(torch.nn.ReLU())
            head_layers.append(torch.nn.Dropout(self.trainer.params.head_dropout))
        head_layers.append(torch.nn.Linear(head_dims[-1], num_labels))
        self.head = torch.nn.Sequential(*head_layers)

        self.loss = torch.nn.CrossEntropyLoss(
            label_smoothing=self.trainer.params.label_smoothing
        )

    def create_optimizer(self):
        return torch.optim.Adam(self.parameters(),
            lr=(self.lr or self.trainer.params.learning_rate)
        )

    def get_node_features_slim(self,
        batch: list[awe.data.graph.dom.Node],
        attrs: bool = True
    ):
        """
        Features used for a node, its ancestor chain, and its visual neighbors.
        """
        x = None

        # Embed HTML tag names.
        if self.html_tag is not None:
            tag_ids = self.html_tag.compute(batch) # [N]
            x = append(x, self.tag_embedding(tag_ids)) # [N, embedding_dim]

        # Tokenize node attributes.
        if attrs and self.trainer.params.tokenize_node_attrs:
            word_ids = self.word_ids.compute_attr(batch) # [N, num_words]
            word_embeddings = self.word_embedding(word_ids)
                # [N, num_words, word_dim]
            attrs = torch.sum(word_embeddings, dim=1) # [N, word_dim]
            x = append(x, attrs)

        return x

    def get_node_features(self, batch: list[awe.data.graph.dom.Node]):
        """Features used for a node and its visual neighbors."""

        x = self.get_node_features_slim(batch,
            attrs=not self.trainer.params.tokenize_node_attrs_only_ancestors
        )

        # Add more node features.
        if self.position is not None:
            x = append(x, self.position.compute(batch))
        if self.visuals is not None:
            x = append(x, self.visuals.compute(batch))

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

    def propagate_visual_neighbors(self, batch: list[awe.data.graph.dom.Node]):
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

        # Normalize coefficients.
        coefficients = coefficients.reshape((len(batch), 1, n_neighbors))
            # [N, 1, n_neighbors]
        normalize = self.trainer.params.neighbor_normalize
        N = awe.training.params.AttentionNormalization
        if normalize == N.vector:
            coefficients = F.normalize(coefficients, dim=-1)
        elif normalize == N.softmax:
            coefficients = self.neighbor_softmax(coefficients)

        # Aggregate neighbor features (sum weighted by the coefficients).
        neighbor_features = neighbor_features \
            .reshape((len(batch), n_neighbors, -1))
            # [N, n_neighbors, node_features]
        neighborhood = torch.matmul(coefficients, neighbor_features)
            # [N, 1, node_features]
        neighborhood = neighborhood.reshape((len(batch), -1))
            # [N, node_features]

        neighborhood = self.dropout(neighborhood)

        return torch.concat((node_features, neighborhood), dim=-1)
            # [N, 2 * node_features]

    def get_ancestor_chain(self, batch: list[awe.data.graph.dom.Node]):
        # Get features for each ancestor.
        n_ancestors = self.trainer.params.n_ancestors
        ancestors = [
            ancestor
            for node in batch
            for ancestor in node.get_ancestor_chain(n_ancestors)
        ] # [N * n_ancestors]
        ancestor_features = self.get_node_features_slim(ancestors)
            # [N * n_ancestors, slim_dim]

        # Append distance of ancestors.
        if self.ancestor_dist:
            ancestor_distances = torch.arange(
                n_ancestors, 0, -1,
                device=self.trainer.device
            ).expand((len(batch), n_ancestors)).flatten() # [N * n_ancestors]
            ancestor_distances = ancestor_distances.reshape((-1, 1))
                # [N * n_ancestors, 1]
            ancestor_features = torch.concat(
                (ancestor_distances, ancestor_features),
                dim=-1
            ) # [N * n_ancestors, slim_dim + 1]

        # Summarize ancestor features.
        if self.trainer.params.ancestor_summarize:
            ancestor_features = self.ancestor_layer(ancestor_features)
                # [N * n_ancestors, anc_dim]
            ancestor_features = self.dropout(ancestor_features)
            ancestor_features: torch.Tensor

        # Split ancestors according to the node they correspond to.
        ancestor_features = ancestor_features.reshape(
            (len(batch), n_ancestors, -1)) # [N, n_ancestors, anc_dim]
        ancestor_features = ancestor_features.transpose(0, 1)
            # [n_ancestors, N, anc_dim]

        # Aggregate ancestor chains.
        if self.trainer.params.ancestor_function == 'lstm':
            # Use LSTM.
            ancestor_chains, _ = self.ancestor_lstm(ancestor_features)
                # [n_ancestors, N, D * anc_out_dim]

            # Aggregate across ancestors.
            ancestor_chains = torch.mean(ancestor_chains, dim=0)
                # [N, D * anc_out_dim]

            ancestor_chains = self.dropout(ancestor_chains)
        else:
            # Use simple aggregation function (sum or mean).
            f = getattr(torch, self.trainer.params.ancestor_function)
            ancestor_chains = f(ancestor_features, dim=0)
                # [N, anc_dim]

        return ancestor_chains

    def forward(self, batch: ModelInput) -> ModelOutput:
        # Propagate visual neighbors.
        if self.trainer.params.visual_neighbors:
            x = self.propagate_visual_neighbors(batch)
        else:
            x = self.get_node_features(batch)

        # Append ancestor chain.
        if self.trainer.params.ancestor_chain:
            x = append(x, self.get_ancestor_chain(batch))

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
