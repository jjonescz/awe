import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Type, TypeVar

import torch

if TYPE_CHECKING:
    from awe import awe_graph

T = TypeVar('T', bound='Feature') # pylint: disable=invalid-name

class FeatureContext:
    """Everything needed to compute a `HtmlNode`'s `Feature`s."""
    page: 'awe_graph.HtmlPage'

    _nodes: list['awe_graph.HtmlNode'] = None

    def __init__(self, page: 'awe_graph.HtmlPage'):
        self.page = page

    @property
    def nodes(self):
        """Cached list of `page.nodes`."""
        if self._nodes is None:
            self._nodes = list(self.page.nodes)
        return self._nodes

    def add(self, feature: Type['Feature']):
        for node in self.nodes:
            feature.add_to(node, self)

    def add_all(self, features: Iterable[Type['Feature']]):
        for feature in features:
            self.add(feature)

class Feature(ABC):
    @property
    @abstractmethod
    def result_len(self) -> int:
        """Length of the resulting feature vector."""

    @abstractmethod
    def create(self,
        node: 'awe_graph.HtmlNode',
        context: FeatureContext) -> torch.FloatTensor:
        """Computes feature vector for the given `node`."""

@dataclass
class Depth(Feature):
    """Relative depth of node in DOM tree."""

    @property
    def result_len(self):
        return 1

    @staticmethod
    def _get_max_depth(context: FeatureContext):
        max_depth = getattr(context, 'max_depth', None)
        if max_depth is None:
            max_depth = max(map(lambda n: n.depth, context.nodes))
            setattr(context, 'max_depth', max_depth)
        return max_depth

    def create(self, node: 'awe_graph.HtmlNode', context: FeatureContext):
        return torch.FloatTensor([node.depth / self._get_max_depth(context)])

@dataclass
class CharCategories(Feature):
    """Counts of different character categories."""

    @property
    def result_len(self):
        return 3

    def create(self, node: 'awe_graph.HtmlNode', _):
        def count_pattern(pattern: str):
            return len(re.findall(pattern, node.text)) if node.is_text else 0

        return torch.FloatTensor([
            count_pattern(r'[$]'),
            count_pattern(r'[a-zA-Z]'),
            count_pattern(r'\d')
        ])
