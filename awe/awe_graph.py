from dataclasses import dataclass, field
from typing import Union

import parsel
from lxml import etree

from awe import html_utils, utils


class HtmlLabels:
    def get_labels(self, node: 'HtmlNode') -> list[str]:
        raise NotImplementedError()

class HtmlPage:
    @property
    def dom(self) -> parsel.Selector:
        raise NotImplementedError()

    @property
    def labels(self) -> HtmlLabels:
        raise NotImplementedError()

    @property
    def root(self):
        # Prepare page DOM.
        page_dom = self.dom
        html_utils.clean(page_dom)

        return HtmlNode(self, 0, page_dom.root)

    @property
    def nodes(self):
        page_labels = self.labels
        deep_index = 0
        for node in self.root.descendants:
            # Exclude whitespace fragments.
            if node.is_text and node.text.isspace():
                continue

            # Create node representation.
            node.labels = page_labels.get_labels(node)
            node.deep_index = deep_index

            yield node
            deep_index += 1

@dataclass
class HtmlNode:
    page: HtmlPage = field(repr=False)

    deep_index: int = field(init=False, default=0)
    """Iteration index of the node inside the `page`."""

    index: int
    """Index inside `parent`."""

    element: Union[etree._Element, str]
    """Node or text fragment."""

    parent: Union['HtmlNode', None] = field(default=None)

    labels: list[str] = field(init=False, default_factory=list)
    """
    Ground-truth labels of the node or `[]` if it doesn't correspond to any
    target attribute.
    """

    _children: list['HtmlNode'] = utils.cache_field()

    @property
    def is_text(self):
        return isinstance(self.element, str)

    @property
    def text(self) -> str:
        assert self.is_text
        return self.element

    @property
    def xpath(self):
        if self.is_text:
            return f'{self.parent.xpath}/text()[{self.index + 1}]'
        return html_utils.get_el_xpath(self.element)

    @property
    def children(self):
        if self._children is None:
            self._children = list(self._iterate_children())
        return self._children

    def _iterate_children(self):
        if not self.is_text:
            index = 0

            if self.element.text is not None:
                yield HtmlNode(self.page, index, self.element.text, self)
                index += 1

            for child in self.element:
                child: etree._Element
                yield HtmlNode(self.page, index, child, self)
                index += 1

                if child.tail is not None:
                    yield HtmlNode(self.page, index, child.tail, self)
                    index += 1

    @property
    def descendants(self):
        stack = [self]
        while len(stack) != 0:
            node = stack.pop()
            yield node
            stack.extend(node.children)
