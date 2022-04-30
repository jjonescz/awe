import abc
from typing import Optional

import awe.data.graph.dom
import awe.data.parsing
import awe.data.set.pages


class PageLabels(abc.ABC):
    """Gold page labels."""

    def __init__(self, page: awe.data.set.pages.Page):
        self.page = page

    @property
    @abc.abstractmethod
    def label_keys(self) -> list[str]:
        """
        List of label names that could be present on the `page` (e.g., `name`,
        `price`).
        """

    # pylint: disable-next=no-self-use,unused-argument
    def get_selector(self, label_key: str) -> Optional[str]:
        """
        CSS selector that can be used to find gold nodes for the specified
        `label_key` if provided in the dataset.
        """
        return None

    @abc.abstractmethod
    def get_label_values(self, label_key: str) -> list[str]:
        """
        Groundtruth text values of the label with the specified `label_key` on
        the `page`.
        """

    @abc.abstractmethod
    def get_labeled_nodes(self, label_key: str) -> list[awe.data.parsing.Node]:
        """
        Groundtruth DOM nodes that are labeled with the specified `label_key`.
        """

class TextPageLabels(PageLabels):
    """When the label's value is given as text."""

    def get_labeled_nodes(self, label_key: str):
        return [
            node
            for value in self.get_label_values(label_key)
            for node in awe.data.parsing.find_nodes_with_text(
                self.page.dom.tree, value
            )
        ]
