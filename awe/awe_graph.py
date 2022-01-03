from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Union

import parsel
from lxml import etree

from awe import html_utils, utils
from awe.data.wayback import WaybackPage

if TYPE_CHECKING:
    from awe import features, visual


class HtmlLabels(ABC):
    @abstractmethod
    def get_labels(self, node: 'HtmlNode') -> list[str]:
        pass

    @abstractmethod
    def get_nodes(self,
        label: str,
        ctx: 'features.PageContextBase'
    ) -> list['HtmlNode']:
        pass

class HtmlPage(ABC):
    archived: Optional[WaybackPage] = False
    """
    `False` means the WaybackMachine API was not called yet;
    `None` means it was called and returned no snapshots.
    """

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Unique identifier of the page (e.g., relative path in dataset)."""

    @property
    @abstractmethod
    def relative_original_path(self) -> str:
        """
        Path relative to dataset directory, without any suffix.

        Useful as an input to our visual attribute extractor utility.
        """

    @property
    @abstractmethod
    def group_key(self) -> str:
        """Identifier of page group (e.g., folder in dataset)."""

    @property
    @abstractmethod
    def group_index(self) -> int:
        """Index inside group determined by `group_key`."""

    @property
    @abstractmethod
    def dom(self) -> parsel.Selector:
        pass

    @property
    def has_dom_data(self) -> bool:
        return False

    @property
    @abstractmethod
    def labels(self) -> HtmlLabels:
        pass

    @property
    @abstractmethod
    def fields(self) -> list[str]:
        """Names of labels recognized in this page."""

    @property
    def data_point_path(self) -> Optional[str]:
        """
        Path to a file with stored Pytorch Geometric datapoints.

        If `None`, the data will be stored in memory.
        """
        return None

    @abstractmethod
    def count_label(self, label: str) -> int:
        """Number of occurrences of nodes with the given `label`."""

    # pylint: disable-next=unused-argument
    def get_groundtruth_texts(self, label: str) -> Optional[list[str]]:
        """Groundtruth texts (if available)"""
        return None

    def create_root(self):
        return HtmlNode(self, 0, 0, self.dom.root)

    def initialize_tree(self):
        root = self.create_root()
        page_labels = self.labels
        deep_index = 0
        for node in root.descendants:
            node.deep_index = deep_index

            # Find groundtruth labels for the node.
            node.labels = page_labels.get_labels(node)

            deep_index += 1
        return root

    def prepare(self, ctx: 'features.PageContextBase'):
        """Prepare page features."""

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def center_point(self):
        return self.x + self.width / 2, self.y + self.height / 2

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

    font_size: Optional[float] = utils.lazy_field()
    """In pixels."""

    font_weight: Optional[float] = utils.lazy_field()
    """In font weight units divided by 100. E.g., "normal" is 4."""

    text_transform: Optional[str] = utils.lazy_field()

    font_style: Optional[str] = utils.lazy_field()

    text_align: Optional[str] = utils.lazy_field()

    cursor: Optional[str] = utils.lazy_field()

    letter_spacing: Optional[float] = utils.lazy_field()
    """In pixels."""

    line_height: Optional[float] = utils.lazy_field()
    """In pixels."""

    opacity: Optional[float] = utils.lazy_field()
    """0 = transparent, 1 = opaque."""

    overflow: Optional[str] = utils.lazy_field()

    pointer_events: Optional[str] = utils.lazy_field()

    text_overflow: Optional[str] = utils.lazy_field()

    color: Optional['visual.Color'] = utils.lazy_field()

    background_color: Optional['visual.Color'] = utils.lazy_field()

    text_decoration: Optional[str] = utils.lazy_field()

    background_image: Optional[str] = utils.lazy_field()

    box_shadow: Optional[str] = utils.lazy_field()

    text_shadow: Optional[str] = utils.lazy_field()

    z_index: Optional[str] = utils.lazy_field()

    _children: list['HtmlNode'] = utils.cache_field()

    @property
    def is_text(self):
        return isinstance(self.element, str)

    @property
    def text(self) -> str:
        assert self.is_text
        return self.element

    @property
    def tag_name(self) -> str:
        if self.is_text:
            return '#text'
        return self.element.tag

    @property
    def summary(self):
        if self.is_text:
            return f'"{self.text}"'
        element_id = self.element.get('id')
        id_str = f'#{element_id}' if element_id is not None else ''
        return f'<{self.tag_name}{id_str}>{len(self.children)}'

    def _get_xpath(self,
        node_filter: Callable[['HtmlNode'], bool] = lambda _: True
    ):
        if self.is_text:
            xpath = f'{self.parent.xpath}/text()'
            num_text_siblings = sum(map(lambda _: 1, filter(
                lambda n: n.is_text and node_filter(n), self.parent.children)))
            if num_text_siblings > 1:
                # Append index only if there are multiple text nodes.
                num_text_prev_siblings = sum(map(lambda _: 1, filter(
                    lambda n: n.is_text and node_filter(n),
                    self.prev_siblings
                )))
                xpath += f'[{num_text_prev_siblings + 1}]'
            return xpath
        return html_utils.get_el_xpath(self.element)

    @property
    def xpath_swde(self):
        """XPath without white-space-only text fragments."""
        # See AWE-1.
        return self._get_xpath(
            lambda n: not n.element.isspace() or '\n' in n.element
        )

    @property
    def xpath(self):
        return self._get_xpath()

    @property
    def children(self):
        if self._children is None:
            child_depth = self.depth + 1
            self._children = [
                HtmlNode(self.page, index, child_depth, child, self)
                for index, child in enumerate(self._iterate_children())
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

    def iterate_descendants(self,
        shallow_predicate: Callable[['HtmlNode'], bool],
        deep_predicate: Callable[['HtmlNode'], bool]
    ):
        stack = [self]
        while len(stack) != 0:
            node = stack.pop()
            if shallow_predicate(node):
                yield node
            if deep_predicate(node):
                stack.extend(node.children)

    @property
    def descendants(self):
        return self.iterate_descendants(lambda _: True, lambda _: True)

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

    @property
    def visual_node(self):
        """
        Parent element for text fragments (because they don't have most of
        visual features by themselves).
        """
        if self.is_text:
            return self.parent
        return self

    def copy_visual_features(self):
        """
        Copies visual features from parent element for text fragments (because
        they don't have most of visual features by themselves).
        """
        if self.is_text:
            self.font_family = self.parent.font_family
            self.font_size = self.parent.font_size
            self.font_weight = self.parent.font_weight
            self.text_transform = self.parent.text_transform
            self.font_style = self.parent.font_style
            self.text_align = self.parent.text_align
            self.cursor = self.parent.cursor
            self.letter_spacing = self.parent.letter_spacing
            self.line_height = self.parent.line_height
            self.opacity = self.parent.opacity
            self.overflow = self.parent.overflow
            self.pointer_events = self.parent.pointer_events
            self.text_overflow = self.parent.text_overflow
            self.color = self.parent.color
            self.background_color = self.parent.background_color
            self.text_decoration = self.parent.text_decoration
            self.background_image = self.parent.background_image
            self.box_shadow = self.parent.box_shadow
            self.text_shadow = self.parent.text_shadow
            self.z_index = self.parent.z_index
