from dataclasses import dataclass, field
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

        parent_node, parent_element, prev_node, prev_element = [None] * 4
        index = 0
        for xpath, element in html_utils.iter_with_fragments(page_dom.root):
            is_text = isinstance(element, str)

            # Exclude whitespace fragments.
            if is_text and element.isspace():
                continue

            # Check relations.
            go_up, go_down = [False] * 2
            if not is_text:
                curr_parent_element = element.getparent()
                if curr_parent_element != parent_element:
                    # Parent DOM element has changed.
                    if curr_parent_element == prev_element:
                        # If current parent is the previous element, it means
                        # the iteration went to children of the previous
                        # element.
                        go_down = True
                    else:
                        # Otherwise, the iteration must have gone up.
                        go_up = True
            else:
                current_parent_xpath = html_utils.get_parent_xpath(xpath)
                if current_parent_xpath != parent_node.xpath:
                    # Parent XPath doesn't match.
                    if parent_node.xpath.startswith(current_parent_xpath):
                        # If parent's XPath still holds, the iteration went to
                        # child text fragments of the previous element.
                        go_down = True
                    else:
                        # Otherwise, the iteration must have gone up.
                        go_up = True
            if go_down:
                parent_element = prev_element
                parent_node = prev_node
            elif go_up:
                grandparent = parent_element.getparent()
                assert curr_parent_element == grandparent
                parent_element = grandparent
                parent_node = parent_node.parent

            # Create node representation.
            labels = page_labels.get_labels(xpath)
            text = element if is_text else None
            node = HtmlNode(self, index, xpath, labels, text, parent_node)

            # Update parent's children.
            if parent_node is not None:
                parent_node.children.append(node)

            yield node
            index += 1
            prev_node = node
            prev_element = element

@dataclass
class HtmlNode:
    page: HtmlPage = field(repr=False)

    index: int
    """Iteration index of the node inside the `page`."""

    xpath: str

    labels: list[str]
    """
    Ground-truth labels of the node or `[]` if it doesn't correspond to any
    target attribute.
    """

    text: Union[str, None]
    """Content if this node is a text fragment; otherwise, `None`."""

    parent: Union['HtmlNode', None]

    children: list['HtmlNode'] = field(default_factory=list)
