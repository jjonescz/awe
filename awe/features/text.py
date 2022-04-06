import itertools
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

    node_attr_token_ids: dict[awe.data.graph.dom.Node, torch.Tensor]
    """
    Cache of node token IDs of attributes values (`id`, `name`, `class`, etc.).
    """

    def get_pickled_keys(self):
        return ('max_num_words',)

    def __post_init__(self, restoring: bool):
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
        self.node_attr_token_ids = {}
        self.default_token_ids = torch.zeros(0,
            dtype=torch.int32,
            device=self.trainer.device
        )

    def tokenize(self, text: str, humanize: bool = False):
        if humanize:
            text = humanize_string(text)
        return self._tokenize(text)

    def get_token_id(self, token: str):
        # Indices start at 1; 0 is used for unknown and pad words.
        return self.glove.get_index(token, default=-1) + 1

    def prepare(self, node: awe.data.graph.dom.Node, train: bool):
        params = self.trainer.params

        # Find maximum word count and save token IDs.
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

        # Tokenize attribute values.
        if self.trainer.params.tokenize_node_attrs:
            attr_text = get_node_attr_text(node)
            if attr_text:
                token_ids = list(itertools.islice(
                    (
                        token_id
                        for token in self.tokenize(attr_text, humanize=True)
                        if (token_id := self.get_token_id(token)) != 0
                    ),
                    0, params.attr_cutoff_words
                ))
                self.node_attr_token_ids[node] = torch.tensor(token_ids,
                    dtype=torch.int32,
                    device=self.trainer.device
                )

    def compute(self, batch: list[list[awe.data.graph.dom.Node]]):
        # Get word token indices.
        return torch.nn.utils.rnn.pack_sequence(
            [
                torch.tensor(
                    [
                        token_id
                        for node in row
                        for token_id in self.node_token_ids[node]
                    ]
                    if len(row) > 0 else [0],
                    dtype=torch.int32,
                    device=self.trainer.device,
                )
                for row in batch
            ],
            enforce_sorted=False
        )

    def compute_attr(self, batch: list[list[awe.data.graph.dom.Node]]):
        return torch.nn.utils.rnn.pad_sequence(
            [
                self.node_attr_token_ids.get(n, self.default_token_ids)
                for n in batch
            ],
            batch_first=True
        )

def get_node_attr_text(node: awe.data.graph.dom.Node):
    attrs = node.get_attributes()
    return ' '.join(
        v
        for a in ['itemprop', 'id', 'name', 'class']
        if (v := attrs.get(a, ''))
    )
