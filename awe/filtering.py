from abc import ABC, abstractmethod
from typing import Any

from lxml import etree

from awe import awe_graph
from awe.features.visual import dom_data


class NodePredicate(ABC):
    def include_node_descendants(self, node: awe_graph.HtmlNode) -> bool:
        """Whether `node`'s descendants should be included in page nodes."""
        return self.include_node(node)

    def include_node_itself(self, node: awe_graph.HtmlNode) -> bool:
        """Whether `node` should be included in page nodes."""
        return self.include_node(node)

    @abstractmethod
    def include_node(self, node: awe_graph.HtmlNode) -> bool:
        """
        Whether `node` and its descendants should be included in page nodes.
        """

    @abstractmethod
    def include_visual(self, node_data: dict[str, Any], node_name: str) -> bool:
        """Whether visual data of node's descendants should be imported."""

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
            dom_data.get_tag_name(node_name) not in self.ignored_tag_names
        )

class LeafNodePredicate(DefaultNodePredicate):
    """Keeps only leaf (text) nodes."""

    def include_node_itself(self, node: awe_graph.HtmlNode):
        if not node.is_text:
            return False
        return super().include_node(node)

    def include_visual(self, node_data: dict[str, Any], node_name: str):
        if dom_data.get_tag_name(node_name) != 'text':
            return False
        return super().include_visual(node_data, node_name)
