from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type, TypeVar

from awe import awe_graph

T = TypeVar('T', bound='Feature') # pylint: disable=invalid-name

class Feature(ABC):
    @abstractmethod
    def apply_to(self, node: awe_graph.HtmlNode) -> bool:
        pass

    @classmethod
    def add_to(cls: Type[T], node: awe_graph.HtmlNode) -> T:
        feature = cls()
        if feature.apply_to(node):
            node.features.append(feature)

@dataclass
class DollarSigns(Feature):
    count: int = field(init=False)

    def apply_to(self, node: awe_graph.HtmlNode):
        if node.is_text:
            self.count = node.text.count('$')
            return True
        return False
