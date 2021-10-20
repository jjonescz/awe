from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import parsel
from lxml import etree

from awe import html_utils, utils


class HtmlLabels(ABC):
    @abstractmethod
    def get_labels(self, node: 'HtmlNode') -> list[str]:
        pass

class HtmlPage(ABC):
    @property
    @abstractmethod
    def dom(self) -> parsel.Selector:
        pass

    @property
    @abstractmethod
    def labels(self) -> HtmlLabels:
        pass

    @property
    @abstractmethod
    def fields(self) -> list[str]:
        """Names of labels recognized in this page."""

    @property
    def root(self):
        # Prepare page DOM.
        page_dom = self.dom
        html_utils.clean(page_dom)

        return HtmlNode(self, 0, 0, page_dom.root)

    @property
    def nodes(self):
        page_labels = self.labels
        deep_index = 0
        for node in self.root.descendants:
            # Create node representation.
            node.labels = page_labels.get_labels(node)
            node.deep_index = deep_index

            yield node
            deep_index += 1

@dataclass
class HtmlNode:
    page: HtmlPage = field(repr=False)

    deep_index: Union[int, None] = field(init=False, default=None)
    """Iteration index of the node inside the `page`."""

    index: int
    """Index inside `parent`."""

    depth: int
    """Level of nesting."""

    element: Union[etree._Element, str]
    """Node or text fragment."""

    parent: Union['HtmlNode', None] = field(repr=False, default=None)

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
            child_depth = self.depth + 1
            self._children = [
                HtmlNode(self.page, index, child_depth, child, self)
                for index, child in enumerate(
                    child for child in self._iterate_children()
                    # Exclude whitespace nodes.
                    if not isinstance(child, str) or not child.isspace()
                )
            ]
        return self._children

    def _iterate_children(self):
        if not self.is_text:
            if self.element.text is not None:
                yield self.element.text

            for child in self.element:
                child: etree._Element
                yield child

                if child.tail is not None:
                    yield child.tail

    @property
    def descendants(self):
        stack = [self]
        while len(stack) != 0:
            node = stack.pop()
            yield node
            stack.extend(node.children)

    @property
    def prev_siblings(self):
        return self.parent.children[:self.index]

    @property
    def next_siblings(self):
        return self.parent.children[self.index + 1:]

    @property
    def siblings(self):
        return self.prev_siblings + self.next_siblings

    @property
    def text_content(self) -> str:
        if self.is_text:
            return self.text
        return self.element.text_content()
