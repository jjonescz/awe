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
            tokenizer = awe.features.bert_tokenization.BasicTokenizer(
                split_on_symbol=(params.tokenizer_id == 'symbol')
            )
            self._tokenize = tokenizer.tokenize

        self.glove = awe.data.glove.LazyEmbeddings.get_or_create()
        self.default_token_ids = torch.tensor([0],
            dtype=torch.int32,
            device=self.trainer.device
        )
        self.enable_cache()

    def tokenize(self, text: str, humanize: bool = False):
        """Obtains list of tokens from `text` using the configured tokenizer."""

        if humanize:
            text = humanize_string(text)
        return self._tokenize(text)

    def get_token_id(self, token: str):
        """
        Transforms textual token to its dictionary ID.

        IDs start at 1; 0 is used for unknown and pad tokens.
        """

        return self.glove.get_index(token, default=-1) + 1

    def compute_node_token_ids(self, node: awe.data.graph.dom.Node, train: bool):
        """Computes list of token IDs for `node`'s textual content."""

        params = self.trainer.params

        # Find maximum word count and save token IDs.
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
        return token_ids

    def compute_node_attr_token_ids(self, node: awe.data.graph.dom.Node):
        """Computes list of token IDs for selected `node`'s attribute values."""

        params = self.trainer.params

        # Tokenize attribute values.
        if (attr_text := self.get_node_attr_text(node)):
            token_ids = list(itertools.islice(
                (
                    token_id
                    for token in self.tokenize(attr_text, humanize=True)
                    if (token_id := self.get_token_id(token)) != 0
                ),
                0, params.attr_cutoff_words
            ))
            return torch.tensor(token_ids,
                dtype=torch.int32,
                device=self.trainer.device
            )
        return self.default_token_ids

    def prepare(self, node: awe.data.graph.dom.Node, train: bool):
        """If cache is enabled, token IDs are cached for the `node`."""

        if self.node_token_ids is not None and node.is_text:
            self.node_token_ids[node] = self.compute_node_token_ids(
                node=node,
                train=train
            )

        if (
            self.node_attr_token_ids is not None and
            self.trainer.params.tokenize_node_attrs
        ):
            self.node_attr_token_ids[node] = self.compute_node_attr_token_ids(
                node=node
            )

    def enable_cache(self, enable: bool = True):
        if enable:
            self.node_token_ids = {}
            self.node_attr_token_ids = {}
        else:
            self.node_token_ids = None
            self.node_attr_token_ids = None

    def compute(self, batch: list[list[awe.data.graph.dom.Node]]):
        """Computes (or retrieves cached) token IDs for all nodes in `batch`."""

        # Get word token indices.
        return torch.nn.utils.rnn.pack_sequence(
            [
                torch.tensor(
                    [
                        token_id
                        for node in row
                        if node.is_text
                        for token_id in (
                            self.node_token_ids[node]
                            if self.node_token_ids is not None
                            else self.compute_node_token_ids(node, train=False)
                        )
                    ] or [0],
                    dtype=torch.int32,
                    device=self.trainer.device,
                )
                for row in batch
            ],
            enforce_sorted=False
        )

    def compute_attr(self, batch: list[awe.data.graph.dom.Node]):
        """
        Like `compute` but for node DOM attribute values instead of textual
        content.
        """

        return torch.nn.utils.rnn.pack_sequence(
            [
                (
                    self.node_attr_token_ids.get(node) or self.default_token_ids
                    if self.node_attr_token_ids is not None
                    else self.compute_node_attr_token_ids(node)
                )
                for node in batch
            ],
            enforce_sorted=False
        )

    def get_node_attr_text(self, node: awe.data.graph.dom.Node):
        """Convenience wrapper for static method `get_node_attr_text`."""

        return get_node_attr_text(node=node, params=self.trainer.params)

def get_node_attr_text(
    node: awe.data.graph.dom.Node,
    params: awe.training.params.Params
):
    """
    Obtains attribute values of `node` as normal space-separated text that can
    be tokenized as usual.
    """

    attrs = node.get_attributes()
    return ' '.join(
        v
        for a in params.tokenize_node_attrs
        if (v := attrs.get(a, ''))
    )
