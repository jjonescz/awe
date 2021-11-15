from abc import ABC, abstractmethod
from typing import Any

from lxml import etree

from awe import awe_graph, visual

class NodePredicate(ABC):
    @abstractmethod
    def include_node(self, node: awe_graph.HtmlNode) -> bool:
        """Whether `node` should be included in page nodes."""

    @abstractmethod
    def include_visual(self, node_data: dict[str, Any], node_name: str) -> bool:
        """Whether visual data should be imported."""

class DefaultNodePredicate(NodePredicate):
    """Ignores whitespace-only nodes, comments and script/style tags."""

    ignored_tag_names = ['script', 'style', 'noscript', 'iframe']

    def include_node(self, node: awe_graph.HtmlNode):
        if node.is_text:
            return not node.text.isspace()
        return not (
            node.element.tag is etree.Comment or
            node.element.tag in self.ignored_tag_names
        )

    def include_visual(self, node_data: dict[str, Any], node_name: str):
        return (
            node_data.get('whiteSpace') is not True and
            visual.get_tag_name(node_name) not in self.ignored_tag_names
        )
