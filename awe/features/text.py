import functools
import re
from typing import TYPE_CHECKING

import torch
from torchtext.data import utils as text_utils
import transformers

import awe.features.feature
import awe.data.glove
import awe.data.graph.dom
import awe.training.params

if TYPE_CHECKING:
    import awe.features.extraction
    import awe.model.classifier


# Inspired by `torchtext.data.utils._basic_english_normalize`.
patterns = [r'\'',
            r'\"',
            r'\.',
            r'<br \/>',
            r',',
            r'\(',
            r'\)',
            r'\!',
            r'\?',
            r'\;',
            r'\:',
            r'\s+',
            r'\$']
replacements = [' \'  ',
                '',
                ' . ',
                ' ',
                ' , ',
                ' ( ',
                ' ) ',
                ' ! ',
                ' ? ',
                ' ',
                ' ',
                ' ',
                ' $ ']
patterns_dict = list((re.compile(p), r) for p, r in zip(patterns, replacements))
def basic_tokenize(line: str):
    line = line.lower()
    for pattern_re, replaced_str in patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()

class WordIdentifiers(awe.features.feature.Feature):
    """Identifiers of word tokens. Used for pre-trained GloVe embeddings."""

    max_num_words: int = 0
    """Number of words in the longest node (up to `cutoff_words`)."""

    def __post_init__(self):
        # Create tokenizer according to config.
        params = self.trainer.params
        family = params.tokenizer_family
        if family == awe.training.params.TokenizerFamily.custom:
            self.tokenize = basic_tokenize
        if family == awe.training.params.TokenizerFamily.torchtext:
            tokenizer = text_utils.get_tokenizer(params.tokenizer_id)
            self.tokenize = functools.partial(basic_tokenize,
                tokenizer=tokenizer
            )
        elif family == awe.training.params.TokenizerFamily.transformers:
            tokenizer = transformers.BertTokenizer.from_pretrained(
                params.tokenizer_id
            )
            self.tokenize = tokenizer.tokenize

        self.glove = awe.data.glove.LazyEmbeddings.get_or_create()

    def get_token_id(self, token: str):
        # Indices start at 1; 0 is used for unknown and pad words.
        return self.glove.get_index(token, default=-1) + 1

    def prepare(self, node: awe.data.graph.dom.Node):
        params = self.trainer.params

        # Find maximum word count.
        if node.is_text:
            counter = 0
            for i, _ in enumerate(self.tokenize(node.text)):
                if (
                    params.cutoff_words is not None and
                    i >= params.cutoff_words
                ):
                    break
                counter += 1
            self.max_num_words = max(self.max_num_words, counter)

    def compute(self, batch: list[list[awe.data.graph.dom.Node]]):
        # Account for friend cycles.
        num_words = self.max_num_words
        if self.trainer.params.friend_cycles:
            num_words *= self.trainer.params.max_friends

        # Get word token indices.
        result = torch.zeros(len(batch), num_words,
            dtype=torch.int32,
            device=self.trainer.device,
        )
        for row_idx, row in enumerate(batch):
            for node_idx, node in enumerate(row):
                for token_idx, token in enumerate(self.tokenize(node.text)):
                    if token_idx >= self.max_num_words:
                        break
                    word_idx = self.max_num_words * node_idx + token_idx
                    result[row_idx, word_idx] = self.get_token_id(token)
        return result
