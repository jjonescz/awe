from typing import TYPE_CHECKING

import torch
from torchtext.data import utils as text_utils

import awe.features.feature
import awe.data.glove
import awe.data.graph.dom

if TYPE_CHECKING:
    import awe.features.extraction
    import awe.model.classifier


class WordIdentifiers(awe.features.feature.Feature):
    """Identifiers of word tokens. Used for pre-trained GloVe embeddings."""

    def __post_init__(self):
        self.tokenizer = text_utils.get_tokenizer('basic_english')
        self.glove = awe.data.glove.LazyEmbeddings.get_or_create()

    def prepare(self, node: awe.data.graph.dom.Node):
        # Find maximum word count.
        if node.is_text:
            counter = 0
            for i, _ in enumerate(self.tokenizer(node.text)):
                if (
                    self.extractor.params.cutoff_words is not None and
                    i >= self.extractor.params.cutoff_words
                ):
                    break
                counter += 1
            self.extractor.context.max_num_words = max(
                self.extractor.context.max_num_words, counter
            )

    def initialize(self):
        return self.extractor.context.max_num_words

    def compute(self, batch: 'awe.model.classifier.ModelInput'):
        # Get word token indices.
        result = torch.zeros(len(batch), self.extractor.context.max_num_words,
            dtype=torch.int32)
        for idx, node in enumerate(batch):
            if node.is_text:
                for i, token in enumerate(self.tokenizer(node.text)):
                    if i >= self.extractor.context.max_num_words:
                        break
                    # Indices start at 1; 0 is used for unknown and pad words.
                    result[idx, i] = self.glove.get_index(token, default=-1) + 1
        return result
