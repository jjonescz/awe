import re
from typing import TYPE_CHECKING

import inflection
import torch
import transformers
from torchtext.data import utils as text_utils

import awe.data.glove
import awe.data.graph.dom
import awe.features.bert_tokenization
import awe.features.feature
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

def humanize_string(text: str):
    """
    Converts symbols (in camelCase, snake_case, etc.) into space-separated
    words.
    """

    text = inflection.underscore(text)
    text = inflection.humanize(text)
    return text

class WordIdentifiers(awe.features.feature.Feature):
    """Identifiers of word tokens. Used for pre-trained GloVe embeddings."""

    max_num_words: int = 0
    """Number of words in the longest node (up to `cutoff_words`)."""

    node_token_ids: dict[awe.data.graph.dom.Node, list[int]]
    """Cache of node tokens IDs."""

    def __post_init__(self):
        # Create tokenizer according to config.
        params = self.trainer.params
        family = params.tokenizer_family
        if family == awe.training.params.TokenizerFamily.custom:
            self._tokenize = basic_tokenize
        if family == awe.training.params.TokenizerFamily.torchtext:
            tokenizer = text_utils.get_tokenizer(params.tokenizer_id)
            self._tokenize = tokenizer
        elif family == awe.training.params.TokenizerFamily.transformers:
            if params.tokenizer_fast:
                cls = transformers.BertTokenizerFast
            else:
                cls = transformers.BertTokenizer
            tokenizer = cls.from_pretrained(params.tokenizer_id)
            self._tokenize = tokenizer.tokenize
        elif family == awe.training.params.TokenizerFamily.bert:
            tokenizer = awe.features.bert_tokenization.BasicTokenizer()
            self._tokenize = tokenizer.tokenize

        self.glove = awe.data.glove.LazyEmbeddings.get_or_create()
        self.node_token_ids = {}

    def tokenize(self, text: str, humanize: bool = False):
        if humanize:
            text = humanize_string(text)
        return self._tokenize(text)

    def get_token_id(self, token: str):
        # Indices start at 1; 0 is used for unknown and pad words.
        return self.glove.get_index(token, default=-1) + 1

    def prepare(self, node: awe.data.graph.dom.Node, train: bool):
        params = self.trainer.params

        # Find maximum word count.
        if node.is_text:
            counter = 0
            token_ids = []
            for i, token in enumerate(self.tokenize(node.text)):
                if train:
                    # For train data, find `max_num_words`.
                    if (
                        params.cutoff_words is not None and
                        i >= params.cutoff_words
                    ):
                        break
                else:
                    if i >= self.max_num_words:
                        break
                token_ids.append(self.get_token_id(token))
                counter += 1
            if train:
                self.max_num_words = max(self.max_num_words, counter)
            self.node_token_ids[node] = token_ids

    def compute(self, batch: list[list[awe.data.graph.dom.Node]]):
        # Account for friend cycles.
        num_words = self.max_num_words
        if self.trainer.params.friend_cycles:
            num_words *= max(1, self.trainer.params.max_friends)

        # Get word token indices.
        result = torch.zeros(len(batch), num_words,
            dtype=torch.int32,
            device=self.trainer.device,
        )
        for row_idx, row in enumerate(batch):
            for node_idx, node in enumerate(row):
                for token_idx, token_id in enumerate(self.node_token_ids[node]):
                    if token_idx >= self.max_num_words:
                        break
                    word_idx = self.max_num_words * node_idx + token_idx
                    result[row_idx, word_idx] = token_id
        return result
