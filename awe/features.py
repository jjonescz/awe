import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, TypeVar

import numpy as np
import torch
from gensim import models as gmodels

from awe.data import glove

if TYPE_CHECKING:
    from awe import awe_graph

T = TypeVar('T', bound='Feature') # pylint: disable=invalid-name

class FeatureContext:
    """Everything needed to compute a `HtmlNode`'s `Feature`s."""
    page: 'awe_graph.HtmlPage'

    max_depth: Optional[int] = None
    """Maximum DOM tree depth; stored by `Depth`."""

    glove: Optional[gmodels.KeyedVectors] = None
    """Pre-trained GloVe weights; stored by `WordEmbedding`."""

    _nodes: list['awe_graph.HtmlNode'] = None

    def __init__(self, page: 'awe_graph.HtmlPage'):
        self.page = page

    @property
    def nodes(self):
        """Cached list of `page.nodes`."""
        if self._nodes is None:
            self._nodes = list(self.page.nodes)
        return self._nodes

class Feature(ABC):
    @property
    @abstractmethod
    def labels(self) -> list[str]:
        """Column names of the resulting feature vector."""

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

class WordEmbedding(Feature):
    """Pre-trained GloVe embedding for each word"""

    @property
    def labels(self):
        # TODO: Doesn't match resulting dimensionality!
        return ['word_embedding']

    @staticmethod
    def _get_glove(context: FeatureContext):
        if context.glove is None:
            context.glove = glove.get_embeddings()
        return context.glove

    def create(self, node: 'awe_graph.HtmlNode', context: FeatureContext):
        glove = self._get_glove(context)
        if not node.is_text:
            vector = None
        else:
            try:
                # TODO: Works only for one-word nodes!
                vector = glove[node.text]
            except KeyError:
                vector = None
        if vector is None:
            vector = np.repeat(0, glove.vector_size)
        return torch.FloatTensor(vector)
