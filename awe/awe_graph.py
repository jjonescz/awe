from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

import parsel
from lxml import etree

from awe import html_utils, utils
from awe.data.wayback import WaybackPage


class HtmlLabels(ABC):
    @abstractmethod
    def get_labels(self, node: 'HtmlNode') -> list[str]:
        pass

class HtmlPage(ABC):
    archived: Optional[WaybackPage] = False
    """
    `False` means the WaybackMachine API was not called yet;
    `None` means it was called and returned no snapshots.
    """

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
            node.deep_index = deep_index

            # Find groundtruth labels for the node.
            node.labels = page_labels.get_labels(node)

            yield node
            deep_index += 1

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

@dataclass
class HtmlNode:
    page: HtmlPage = field(repr=False)

    index: int
    """Index inside `parent`."""

    depth: int
    """Level of nesting."""

    element: Union[etree._Element, str]
    """Node or text fragment."""

    parent: Optional['HtmlNode'] = field(repr=False, default=None)

    labels: list[str] = field(init=False, default_factory=list)
    """
    Ground-truth labels of the node or `[]` if it doesn't correspond to any
    target attribute.
    """

    deep_index: Optional[int] = utils.lazy_field()
    """Iteration index of the node inside the `page`."""

    dataset_index: Optional[int] = utils.lazy_field()
    """Index set and used by `Dataset`."""

    box: Optional[BoundingBox] = utils.lazy_field()

    font_family: Optional[str] = utils.lazy_field()

    font_size: Optional[int] = utils.lazy_field()
    """In pixels."""

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
            xpath = f'{self.parent.xpath}/text()'
            num_text_siblings = sum(map(lambda _: 1,
                filter(lambda n: n.is_text, self.parent.children)))
            if num_text_siblings > 1:
                # Append index only if there are multiple text nodes.
                num_text_prev_siblings = sum(map(lambda _: 1,
                    filter(lambda n: n.is_text, self.prev_siblings)))
                xpath += f'[{num_text_prev_siblings + 1}]'
            return xpath
        return html_utils.get_el_xpath(self.element)

    @property
    def children(self):
        if self._children is None:
            child_depth = self.depth + 1
            self._children = [
                HtmlNode(self.page, index, child_depth, child, self)
                for index, child in enumerate(
                    child for child in self._iterate_children()
                    # HACK: Exclude whitespace nodes. Note that SWDE dataset
                    # removes whitespace before matching groundtruth labels, so
                    # this needs to be done for consistency.
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
        if self.parent is None:
            return []
        return self.parent.children[:self.index]

    @property
    def next_siblings(self):
        if self.parent is None:
            return []
        return self.parent.children[self.index + 1:]

    @property
    def siblings(self):
        return self.prev_siblings + self.next_siblings

    @property
    def text_content(self) -> str:
        if self.is_text:
            return self.text
        return self.element.text_content()

    @property
    def is_white_space(self):
        return self.is_text and self.text.isspace()
