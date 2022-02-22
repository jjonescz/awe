from typing import TYPE_CHECKING

import torch
from torchtext.data import utils as text_utils

import awe.features.feature
import awe.data.glove
import awe.data.graph.dom

if TYPE_CHECKING:
    import awe.features.extraction


class WordIdentifiers(awe.features.feature.Feature):
    """Identifiers of word tokens. Used for pre-trained GloVe embeddings."""

    def __init__(self):
        self.tokenizer = text_utils.get_tokenizer('basic_english')
        self.glove = awe.data.glove.LazyEmbeddings.get_or_create()

    def prepare(self,
        node: awe.data.graph.dom.Node,
        extractor: 'awe.features.extraction.Extractor'
    ):
        # Find maximum word count.
        if node.is_text:
            counter = 0
            for i, _ in enumerate(self.tokenizer(node.text)):
                if (
                    extractor.params.cutoff_words is not None and
                    i >= extractor.params.cutoff_words
                ):
                    break
                counter += 1
            extractor.context.max_num_words = max(
                extractor.context.max_num_words, counter
            )

    def initialize(self, extractor: 'awe.features.extraction.Extractor'):
        return extractor.context.max_num_words

    def compute(self,
        node: awe.data.graph.dom.Node,
        extractor: 'awe.features.extraction.Extractor'
    ):
        # Get word token indices.
        result = torch.zeros(extractor.context.max_num_words, dtype=torch.int32)
        if node.is_text:
            for i, token in enumerate(self.tokenizer(node.text)):
                if i >= extractor.context.max_num_words:
                    break
                # Indices start at 1; 0 is used for unknown and pad words.
                result[i] = self.glove.get_index(token, default=-1) + 1
        return result
