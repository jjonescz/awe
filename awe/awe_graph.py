from dataclasses import dataclass
from typing import Union

import parsel

from . import html_utils


class HtmlLabels:
    def get_labels(self, xpath: str) -> list[str]:
        raise NotImplementedError()

class HtmlPage:
    @property
    def dom(self) -> parsel.Selector:
        raise NotImplementedError()

    @property
    def labels(self) -> HtmlLabels:
        raise NotImplementedError()

    @property
    def nodes(self):
        # Prepare page DOM.
        page_dom = self.dom
        html_utils.clean(page_dom)

        page_labels = self.labels
        nodes = html_utils.iter_with_fragments(page_dom.root)
        for index, (xpath, node) in enumerate(nodes):
            is_text = isinstance(node, str)

            # Exclude whitespace fragments.
            if is_text and node.isspace():
                continue

            # Create node representation.
            labels = page_labels.get_labels(xpath)
            text = node if is_text else None
            yield HtmlNode(self, index, xpath, labels, text)

@dataclass
class HtmlNode:
    page: HtmlPage

    id: int
    """Unique ID of the node inside the `page`."""

    xpath: str

    labels: list[str]
    """
    Ground-truth labels of the node or `[]` if it doesn't correspond to any
    target attribute.
    """

    text: Union[str, None]
    """Content if this node is a text fragment; otherwise, `None`."""
