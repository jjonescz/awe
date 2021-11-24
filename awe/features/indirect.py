from typing import TYPE_CHECKING

import torch
from torchtext.data import utils as text_utils

from awe.data import glove
# pylint: disable=wildcard-import, unused-wildcard-import
from awe.features.context import *
from awe.features.feature import *

if TYPE_CHECKING:
    from awe import awe_graph


def get_default_tokenizer():
    return text_utils.get_tokenizer('basic_english')

class CharIdentifiers(IndirectFeature):
    """Identifiers of characters. Used for randomly-initialized embeddings."""

    def __init__(self):
        self.tokenizer = get_default_tokenizer()

    @property
    def label(self):
        return 'char_identifiers'

    def prepare(self, node: 'awe_graph.HtmlNode', context: RootContext):
        # Find all distinct characters and maximum word length and count.
        if node.is_text:
            counter = 0
            for token in self.tokenizer(node.text):
                context.chars.update(char for char in token)
                context.max_word_len = max(context.max_word_len, len(token))
                counter += 1
            if context.cutoff_words is not None:
                counter = min(context.cutoff_words, counter)
            context.max_num_words = max(context.max_num_words, counter)

    def initialize(self, context: LiveContext):
        # Map all found characters to numbers.
        context.char_dict = { c: i for i, c in enumerate(context.root.chars) }

    def compute(self, node: 'awe_graph.HtmlNode', context: PageContext):
        # Get character indices in each word.
        result = torch.zeros(
            context.root.max_num_words,
            context.root.max_word_len,
            dtype=torch.int32
        )
        if node.is_text:
            for i, token in enumerate(self.tokenizer(node.text)):
                result[i, :len(token)] = torch.IntTensor([
                    context.live.char_dict[char] for char in token
                ])
        return result

class WordIdentifiers(IndirectFeature):
    """Identifiers of word tokens. Used for pre-trained GloVe embeddings."""

    def __init__(self):
        self.tokenizer = get_default_tokenizer()

    @property
    def label(self):
        return 'word_identifiers'

    @property
    def glove(self):
        return glove.LazyEmbeddings.get_or_create()

    def prepare(self, node: 'awe_graph.HtmlNode', context: RootContext):
        # Find maximum word count.
        if node.is_text:
            count = sum(1 for _ in self.tokenizer(node.text))
            if context.cutoff_words is not None:
                count = min(context.cutoff_words, count)
            context.max_num_words = max(context.max_num_words, count)

    def initialize(self, _):
        # Load word vectors.
        _ = self.glove

    def compute(self, node: 'awe_graph.HtmlNode', context: PageContext):
        # Get word token indices.
        result = torch.zeros(context.root.max_num_words, dtype=torch.int32)
        if node.is_text:
            for i, token in enumerate(self.tokenizer(node.text)):
                # Indices start at 1; 0 is used for unknown and pad words.
                result[i] = self.glove.get_index(token, default=-1) + 1
        return result
