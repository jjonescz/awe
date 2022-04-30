"""Text-related custom layers."""

from typing import TYPE_CHECKING
import torch
import torch.nn
import torch.nn.functional as F

import awe.data.glove
import awe.data.graph.dom
import awe.features.text
import awe.model.lstm_utils

if TYPE_CHECKING:
    import awe.model.classifier


class WordEmbedding(torch.nn.Module):
    """Wrapper for word embedding."""

    def __init__(self, parent: 'awe.model.classifier.Model'):
        super().__init__()
        self.trainer = parent.trainer

        # Load pre-trained word embedding layer.
        glove_model = awe.data.glove.LazyEmbeddings.get_or_create()
        embeddings = torch.FloatTensor(glove_model.vectors)
        word_num, word_dim = embeddings.shape
        if self.trainer.params.pretrained_word_embeddings:
            # Add one embedding at the top (for unknown and pad words).
            embeddings = F.pad(embeddings, (0, 0, 1, 0))
            self.word_embedding = torch.nn.Embedding.from_pretrained(
                embeddings,
                padding_idx=0,
                freeze=self.trainer.params.freeze_word_embeddings
            )
        else:
            # Use only word indices of the pre-trained embeddings.
            self.word_embedding = torch.nn.Embedding(word_num, word_dim,
                padding_idx=0
            )
        self.out_dim = word_dim

    def forward(self, word_ids):
        return self.word_embedding(word_ids)

class WordLstm(torch.nn.Module):
    """Wrapper for text LSTM."""

    def __init__(self, parent: 'awe.model.classifier.Model'):
        super().__init__()
        self.trainer = parent.trainer
        word_dim = parent.word_embedding.out_dim

        self.dropout = torch.nn.Dropout(0.3)

        if self.trainer.params.word_vector_function == 'lstm':
            self.lstm = torch.nn.LSTM(word_dim, self.trainer.params.lstm_dim,
                **(self.trainer.params.lstm_args or {})
            )

            self.out_dim = self.trainer.params.lstm_dim
            if self.lstm.bidirectional:
                self.out_dim *= 2
        else:
            self.out_dim = word_dim

    def forward(self, batch: list[list[awe.data.graph.dom.Node]]):
        # Extract word identifiers for the batch.
        word_ids = self.trainer.model.word_ids.compute(batch)

        # Embed words.
        embedded_words = self.trainer.model.word_embedding(word_ids.data)
        embedded_words = self.dropout(embedded_words)
        packed_words = awe.model.lstm_utils.re_pack(word_ids, embedded_words)

        if self.trainer.params.word_vector_function == 'lstm':
            # Run through LSTM.
            word_output, (_, _) = self.lstm(packed_words)
            padded_words, _ = torch.nn.utils.rnn.pad_packed_sequence(word_output)
                # [max_num_words, num_nodes, D * lstm_dim]

            # Aggregate across words.
            node_vectors = torch.mean(padded_words, dim=0)
                # [num_nodes, D * lstm_dim]

            node_vectors = self.dropout(node_vectors)
        else:
            # If not using LSTM, aggregate word embeddings.
            padded_words, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_words)
                # [max_num_words, num_nodes, word_dim]
            f = getattr(torch, self.trainer.params.word_vector_function)
            node_vectors = f(padded_words, dim=0) # [num_nodes, word_dim]

        return node_vectors
