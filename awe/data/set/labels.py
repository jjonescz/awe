import abc

import awe.data.graph.dom
import awe.data.parsing
import awe.data.set.pages


class PageLabels(abc.ABC):
    def __init__(self, page: awe.data.set.pages.Page):
        self.page = page
        self.dom = awe.data.graph.dom.Dom(page, self)

    @property
    @abc.abstractmethod
    def label_keys(self) -> list[str]:
        """
        List of label names that could be present on the `page` (e.g., `name`,
        `price`).
        """

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
                self.dom.tree, value
            )
        ]
