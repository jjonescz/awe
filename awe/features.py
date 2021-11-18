import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, TypeVar

import numpy as np
import torch
from torchtext.data import utils as text_utils

from awe import filtering
from awe.data import glove

if TYPE_CHECKING:
    from awe import awe_graph

T = TypeVar('T', bound='Feature') # pylint: disable=invalid-name

class FeatureContext:
    """Everything needed to compute a `HtmlNode`'s `Feature`s."""
    page: 'awe_graph.HtmlPage'

    max_depth: Optional[int] = None
    """Maximum DOM tree depth. Stored by `Depth`."""

    char_dict: set[str] = set()
    """
    All characters present in processed nodes.
    Stored by `CharacterEmbedding`.
    """

    _nodes: list['awe_graph.HtmlNode'] = None

    def __init__(self,
        page: 'awe_graph.HtmlPage',
        node_predicate: filtering.NodePredicate
    ):
        self.page = page
        self.node_predicate = node_predicate

    @property
    def nodes(self):
        """Cached list of `page.nodes`."""
        if self._nodes is None:
            root = self.page.initialize_tree()
            self._nodes = list(root.iterate_descendants(
                self.node_predicate.include_node))
        return self._nodes

class Feature(ABC):
    @property
    @abstractmethod
    def labels(self) -> list[str]:
        """Column names of the resulting feature vector."""

    @property
    def dimension(self) -> Optional[int]:
        """Length of the feature vector or `None` if unknown."""
        return len(self.labels)

    def initialize(self):
        """Work needed to be done so that this feature can be computed."""

    @abstractmethod
    def create(self,
        node: 'awe_graph.HtmlNode',
        context: FeatureContext) -> torch.FloatTensor:
        """Computes feature vector for the given `node`."""

class Depth(Feature):
    """Relative depth of node in DOM tree."""

    @property
    def labels(self):
        return ['relative_depth']

    @staticmethod
    def _get_max_depth(context: FeatureContext):
        if context.max_depth is None:
            context.max_depth = max(map(lambda n: n.depth, context.nodes))
        return context.max_depth

    def create(self, node: 'awe_graph.HtmlNode', context: FeatureContext):
        return torch.FloatTensor([node.depth / self._get_max_depth(context)])

class IsLeaf(Feature):
    """Whether node is leaf (text) node."""

    @property
    def labels(self):
        return ['is_leaf']

    def create(self, node: 'awe_graph.HtmlNode', _):
        return torch.FloatTensor([node.is_text])

class CharCategories(Feature):
    """Counts of different character categories."""

    @property
    def labels(self):
        return ['dollars', 'letters', 'digits']

    def create(self, node: 'awe_graph.HtmlNode', _):
        def count_pattern(pattern: str):
            return len(re.findall(pattern, node.text)) if node.is_text else 0

        return torch.FloatTensor([
            count_pattern(r'[$]'),
            count_pattern(r'[a-zA-Z]'),
            count_pattern(r'\d')
        ])

class FontSize(Feature):
    """Font size (in pixels)."""

    @property
    def labels(self):
        return ['font_size']

    def create(self, node: 'awe_graph.HtmlNode', _):
        return torch.FloatTensor([node.visual_node.font_size or 0])

class CharEmbedding(Feature):
    """Randomly-initialized character embeddings."""

    @property
    def labels(self):
        return ['char_embedding']

    @property
    def dimension(self):
        return None

    def create(self, node: 'awe_graph.HtmlNode', context: FeatureContext):
        if node.is_text:
            context.char_dict.update(char for char in node.text)
            return torch.IntTensor([ord(char) for char in node.text])
        return torch.IntTensor([])

class WordEmbedding(Feature):
    """Pre-trained GloVe embedding for each word -> averaged to one vector."""

    def __init__(self):
        self.tokenizer = text_utils.get_tokenizer('basic_english')

    @property
    def labels(self):
        return ['word_embedding']

    @property
    def dimension(self):
        return glove.VECTOR_DIMENSION

    @property
    def glove(self):
        return glove.LazyEmbeddings.get_or_create()

    def _embed(self, text: str):
        for token in self.tokenizer(text):
            try:
                yield self.glove[token]
            except KeyError:
                pass

    def _get_vector(self, node: 'awe_graph.HtmlNode'):
        if node.is_text:
            vectors = list(self._embed(node.text))
            if len(vectors) != 0:
                return np.mean(vectors, axis=0)
        return np.repeat(0, self.glove.vector_size)

    def initialize(self):
        # Load word vectors.
        _ = self.glove

    def create(self, node: 'awe_graph.HtmlNode', _):
        return torch.FloatTensor(self._get_vector(node))
