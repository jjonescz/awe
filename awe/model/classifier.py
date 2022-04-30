"""The deep learning model."""

import dataclasses
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn
import torch.nn.functional as F

import awe.data.glove
import awe.data.graph.dom
import awe.features.dom
import awe.features.extraction
import awe.features.text
import awe.features.visual
import awe.model.lstm_utils
import awe.model.word_lstm
import awe.training.params

if TYPE_CHECKING:
    import awe.training.trainer


@dataclasses.dataclass
class ModelOutput:
    """Output of the classifier for one batch of inputs."""

    loss: torch.FloatTensor
    """Loss value."""

    logits: torch.FloatTensor
    """Logit values with shape `[batch_size, label_keys]`."""

    gold_labels: torch.FloatTensor
    """Gold label IDs with shape `[batch_size]`."""

    def get_pred_labels(self):
        """Computes predicted label IDs from `logits`."""

        return torch.argmax(self.logits, dim=-1)

    def get_probabilities(self):
        """Normalizes `logits` to probabilities."""

        return F.softmax(self.logits, dim=-1)

@dataclasses.dataclass
class Prediction:
    """Model's input and the corresponding output."""

    batch: list[awe.data.graph.dom.Node]
    """Input batch of nodes."""

    outputs: ModelOutput
    """Output of the model when given the input `batch`."""

    def filter_nodes(self,
        predicate: Callable[[awe.data.graph.dom.Node], bool]
    ):
        """
        Consistently filters the batch and the output according to the given
        `predicate`.
        """

        mask = [predicate(n) for n in self.batch]
        return Prediction(
            batch=[n for n, m in zip(self.batch, mask) if m],
            outputs=ModelOutput(
                loss=self.outputs.loss,
                logits=self.outputs.logits[mask],
                gold_labels=self.outputs.gold_labels[mask],
            )
        )

class Model(torch.nn.Module):
    """The main classifier model."""

    def __init__(self,
        trainer: 'awe.training.trainer.Trainer',
        lr: Optional[float] = None,
    ):
        super().__init__()
        self.trainer = trainer
        self.lr = lr

        self.dropout = torch.nn.Dropout(0.3)

        # Word embedding (shared for node text and attributes)
        self.word_ids = self.trainer.extractor.get_feature(awe.features.text.WordIdentifiers)
        if self.word_ids is not None:
            self.word_embedding = awe.model.word_lstm.WordEmbedding(self)

        node_feature_dim = 0

        # HTML tag name embedding
        self.html_tag = self.trainer.extractor.get_feature(awe.features.dom.HtmlTag)
        if self.html_tag is not None:
            num_html_tags = len(self.html_tag.html_tag_ids) + 1
            embedding_dim = self.trainer.params.tag_name_embedding_dim
            self.tag_embedding = torch.nn.Embedding(
                num_html_tags,
                embedding_dim
            )
            node_feature_dim += embedding_dim

        # HTML attribute text embedding
        if (
            self.trainer.params.tokenize_node_attrs and
            not self.trainer.params.tokenize_node_attrs_only_ancestors
        ):
            node_feature_dim += self.word_embedding.out_dim

        # Node position
        self.position = self.trainer.extractor.get_feature(awe.features.dom.Position)
        if self.position is not None:
            node_feature_dim += self.position.out_dim

        # Node visuals
        self.visuals = self.trainer.extractor.get_feature(awe.features.visual.Visuals)
        if self.visuals is not None:
            node_feature_dim += self.visuals.out_dim

        # Word LSTM
        if self.trainer.params.word_vector_function is not None:
            self.lstm = awe.model.word_lstm.WordLstm(self)
            out_dim = self.lstm.out_dim
            if self.trainer.params.friend_cycles:
                out_dim *= 3
            node_feature_dim += out_dim

        # Visual neighbors
        if self.trainer.params.visual_neighbors:
            self.neighbor_attention = torch.nn.Linear(
                node_feature_dim * 2 + 3, 1
            )
            self.neighbor_softmax = torch.nn.Softmax(dim=-1)
            head_features = node_feature_dim * 2
        else:
            head_features = node_feature_dim

        # XPath
        if self.trainer.params.xpath:
            xpath_dim = 30
            xpath_out_dim = 10
            num_html_tags = len(self.html_tag.html_tag_ids) + 1
            self.xpath_embedding = torch.nn.Embedding(
                num_html_tags,
                xpath_dim
            )
            self.xpath_lstm = torch.nn.LSTM(
                xpath_dim,
                xpath_out_dim,
                bidirectional=True
            )
            if self.xpath_lstm.bidirectional:
                xpath_out_dim *= 2
            head_features += xpath_out_dim

        # Ancestor chain
        if self.trainer.params.ancestor_chain:
            ancestor_input_dim = 0
            if self.trainer.params.tokenize_node_attrs:
                ancestor_input_dim += self.word_embedding.out_dim
            if self.trainer.params.ancestor_tag_dim is not None:
                self.ancestor_tag_embedding = torch.nn.Embedding(
                    len(self.html_tag.html_tag_ids) + 1,
                    self.trainer.params.ancestor_tag_dim
                )
                ancestor_input_dim += self.trainer.params.ancestor_tag_dim
            ancestor_out_dim = self.trainer.params.ancestor_lstm_out_dim
            self.ancestor_lstm = torch.nn.LSTM(
                ancestor_input_dim,
                ancestor_out_dim,
                **(self.trainer.params.ancestor_lstm_args or {})
            )
            if self.ancestor_lstm.bidirectional:
                ancestor_out_dim *= 2
            head_features += ancestor_out_dim

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
        """Initializes new optimizer for training."""

        return torch.optim.Adam(self.parameters(),
            lr=(self.lr or self.trainer.params.learning_rate),
            weight_decay=self.trainer.params.weight_decay,
        )

    def get_node_attrs(self, batch: list[awe.data.graph.dom.Node]):
        """Computes the DOM attribute feature for `batch`."""

        # Get attr tokens.
        attr_tokens = self.word_ids.compute_attr(batch)

        # Embed attr tokens.
        embedded_attrs = self.word_embedding(attr_tokens.data)
        embedded_attrs = self.dropout(embedded_attrs) # [*, word_dim]

        # Pad.
        packed_attrs = awe.model.lstm_utils.re_pack(
            attr_tokens, embedded_attrs
        )
        padded_attrs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_attrs)
            # [max_num_attrs, num_nodes, word_dim]

        # Aggregate.
        node_vectors = torch.mean(padded_attrs, dim=0)
            # [num_nodes, word_dim]

        node_vectors = self.dropout(node_vectors)
        node_vectors: torch.Tensor

        return node_vectors

    def get_node_features(self, batch: list[awe.data.graph.dom.Node]):
        """Computes features used for a node and its visual neighbors."""

        x = None

        # Embed HTML tag names.
        if self.html_tag is not None:
            tag_ids = self.html_tag.compute(batch) # [N]
            x = append(x, self.tag_embedding(tag_ids)) # [N, embedding_dim]

        # Tokenize node attributes.
        if (
            self.trainer.params.tokenize_node_attrs and
            not self.trainer.params.tokenize_node_attrs_only_ancestors
        ):
            x = append(x, self.get_node_attrs(batch))

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
                    if n.friends is None:
                        raise RuntimeError(
                            f'Node has no friends ({n.get_xpath()!r} in ' +
                            f'{n.dom.page.html_path!r}).'
                        )

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
        """Aggregates features from visual neighbors of nodes in `batch`."""

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
                v.get_visual_distance(
                    normalize=self.trainer.params.normalize_distance
                )
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

    def get_xpath(self, batch: list[awe.data.graph.dom.Node]):
        """Similar to ancestor chain, but simpler."""

        # Get features for each ancestor.
        ancestor_html_tag_ids = torch.nn.utils.rnn.pack_sequence(
            [
                self.html_tag.compute(node.get_all_ancestors())
                for node in batch
            ],
            enforce_sorted=False
        )

        # Embed.
        embedded_ancestors = self.xpath_embedding(ancestor_html_tag_ids.data)
        embedded_ancestors = self.dropout(embedded_ancestors)
        packed_ancestors = awe.model.lstm_utils.re_pack(
            ancestor_html_tag_ids, embedded_ancestors
        )

        # Run through LSTM.
        lstm_output, (_, _) = self.xpath_lstm(packed_ancestors)
        padded_ancestors, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output)
            # [max_num_ancestors, num_nodes, D * lstm_dim]

        # Aggregate.
        node_vectors = torch.mean(padded_ancestors, dim=0)
            # [num_nodes, D * lstm_dim]
        node_vectors = self.dropout(node_vectors)

        return node_vectors

    def get_ancestor_chain(self, batch: list[awe.data.graph.dom.Node]):
        """Aggregates feature of the ancestor chain of each node in `batch`."""

        # Get ancestor chain.
        n_ancestors = self.trainer.params.n_ancestors
        ancestors = [
            (
                node.get_ancestor_chain(n_ancestors) if n_ancestors is not None
                else node.get_all_ancestors()
            )
            for node in batch
        ] # [N, n_ancestors]

        ancestor_tags = None
        ancestor_attrs = None
        ancestor_features = None # [*, tag_dim + word_dim]

        if self.trainer.params.ancestor_tag_dim is not None:
            # Get ancestor HTML tags.
            ancestor_tags = torch.nn.utils.rnn.pack_sequence(
                [
                    self.html_tag.compute(chain)
                    for chain in ancestors
                ],
                enforce_sorted=False
            )

            # Embed ancestor HTML tags.
            embedded_tags = self.ancestor_tag_embedding(ancestor_tags.data)
            embedded_tags = self.dropout(embedded_tags) # [*, tag_dim]

            ancestor_features = append(ancestor_features, embedded_tags)

        if self.trainer.params.tokenize_node_attrs:
            # Get ancestor attribute word vectors (mean-aggregated per node).
            ancestor_attrs = torch.nn.utils.rnn.pack_sequence(
                [
                    self.get_node_attrs(chain) # [chain_len, word_dim]
                    for chain in ancestors
                ],
                enforce_sorted=False
            )

            ancestor_features = append(ancestor_features, ancestor_attrs.data)

        # Pack ancestor features together.
        packed_ancestors = awe.model.lstm_utils.re_pack(
            ancestor_tags or ancestor_attrs, ancestor_features
        )

        # Run through LSTM.
        lstm_output, (_, _) = self.ancestor_lstm(packed_ancestors)
        padded_ancestors, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output)
            # [max_num_ancestors, num_nodes, D * lstm_dim]

        # Aggregate.
        node_vectors = torch.mean(padded_ancestors, dim=0)
            # [num_nodes, D * lstm_dim]

        node_vectors = self.dropout(node_vectors)

        return node_vectors

    def forward(self, batch: list[awe.data.graph.dom.Node]) -> ModelOutput:
        """Classifies nodes in `batch`."""

        # Propagate visual neighbors.
        if self.trainer.params.visual_neighbors:
            x = self.propagate_visual_neighbors(batch)
        else:
            x = self.get_node_features(batch)

        # Append XPath.
        if self.trainer.params.xpath:
            x = append(x, self.get_xpath(batch))

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
    """Safely concatenates features."""

    if x is None:
        return y
    return torch.concat((x, y), dim=-1)
