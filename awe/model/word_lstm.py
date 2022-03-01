from typing import TYPE_CHECKING
import torch
import torch.nn
import torch.nn.functional as F

import awe.data.glove
import awe.data.graph.dom
import awe.features.text

if TYPE_CHECKING:
    import awe.model.classifier


class WordLstm(torch.nn.Module):
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
                freeze=self.trainer.params.freeze_word_vectors
            )
        else:
            # Use only word indices of the pre-trained embeddings.
            self.word_embedding = torch.nn.Embedding(word_num, word_dim,
                padding_idx=0
            )

        if self.trainer.params.word_vector_function == 'lstm':
            self.lstm = torch.nn.LSTM(word_dim, self.trainer.params.lstm_dim,
                batch_first=True,
                **(self.trainer.params.lstm_args or {})
            )

            self.out_dim = self.trainer.params.lstm_dim
            if self.lstm.bidirectional:
                # LSTM output will be twice as long.
                self.out_dim *= 2
        else:
            self.out_dim = word_dim

    def forward(self, batch: list[list[awe.data.graph.dom.Node]]):
        # Extract word identifiers for the batch.
        feat = self.trainer.extractor.get_feature(awe.features.text.WordIdentifiers)
        word_ids = feat.compute(batch)

        if self.trainer.params.filter_node_words:
            # Keep only sentences at leaf nodes.
            masked_word_ids = word_ids[batch.target, :] # [num_masked_nodes, max_num_words]
        else:
            masked_word_ids = word_ids

        # Embed words.
        embedded_words = self.word_embedding(masked_word_ids)
            # [num_masked_nodes, max_num_words, word_dim]

        if self.trainer.params.word_vector_function == 'lstm':
            # Run through LSTM.
            word_output, (_, _) = self.lstm(embedded_words)
                # [num_masked_nodes, max_num_words, D * lstm_dim]

            # Keep only the last word output (whole text representation).
            node_vectors = word_output[:, -1, ...] # [num_masked_nodes, D * lstm_dim]
        else:
            # If not using LSTM, aggregate word embeddings.
            f = getattr(torch, self.trainer.params.word_vector_function)
            node_vectors = f(embedded_words, dim=1) # [num_masked_nodes, word_dim]

        if self.trainer.params.filter_node_words:
            # Expand word vectors to the original shape.
            full_node_vectors = torch.zeros(
                word_ids.shape[0], node_vectors.shape[1],
                device=self.trainer.device
            ) # [num_nodes, lstm_dim]
            full_node_vectors[batch.target] = node_vectors
        else:
            full_node_vectors = node_vectors

        return full_node_vectors
