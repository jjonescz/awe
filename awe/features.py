import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Type, TypeVar

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
    @classmethod
    @abstractmethod
    def apply_to(cls,
        node: 'awe_graph.HtmlNode',
        context: FeatureContext
    ) -> bool:
        pass

    @classmethod
    def add_to(
        cls: Type[T],
        node: 'awe_graph.HtmlNode',
        context: FeatureContext
    ) -> T:
        feature = cls.apply_to(node, context)
        if feature is not None:
            node.features.append(feature)

    @classmethod
    def default(cls: Type[T]) -> T:
        raise NotImplementedError()

@dataclass
class Depth(Feature):
    """Relative depth of node in DOM tree."""

    relative: float

    @classmethod
    def apply_to(cls, node: 'awe_graph.HtmlNode', context: FeatureContext):
        max_depth = getattr(context, 'max_depth', None)
        if max_depth is None:
            max_depth = max(map(lambda n: n.depth, context.nodes))
            setattr(context, 'max_depth', max_depth)
        return Depth(node.depth / max_depth)

@dataclass
class CharCategories(Feature):
    """Numbers of different character categories."""

    dollars: int
    letters: int
    digits: int

    @classmethod
    def apply_to(cls, node: 'awe_graph.HtmlNode', _):
        if node.is_text:
            def count_pattern(pattern: str):
                return len(re.findall(pattern, node.text))
            return CharCategories(
                dollars=node.text.count('$'),
                letters=count_pattern(r'[a-zA-Z]'),
                digits=count_pattern(r'\d')
            )
        return None

    @classmethod
    def default(cls):
        return CharCategories(0, 0, 0)
