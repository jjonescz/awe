from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    @abstractmethod
    def apply_to(self,
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
        feature = cls()
        if feature.apply_to(node, context):
            node.features.append(feature)

@dataclass
class DollarSigns(Feature):
    count: int = field(init=False)

    def apply_to(self, node: 'awe_graph.HtmlNode', _):
        if node.is_text:
            self.count = node.text.count('$')
            return True
        return False
