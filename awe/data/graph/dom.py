import dataclasses
from typing import TYPE_CHECKING, Callable, Optional

import awe.data.graph.pred
import awe.data.parsing
import awe.data.html_utils

if TYPE_CHECKING:
    import awe.data.set.labels
    import awe.data.set.pages


class Dom:
    root: Optional['Node'] = None
    nodes: Optional[list['Node']] = None
    labeled_parsed_nodes: dict[str, list[awe.data.parsing.Node]]
    labeled_nodes: dict[str, list['Node']]

    def __init__(self,
        page: 'awe.data.set.pages.Page'
    ):
        self.page = page
        self.labeled_parsed_nodes = {}
        self.labeled_nodes = {}
        self.tree = awe.data.parsing.parse_html(page.get_html_text())

    def init_nodes(self):
        # Get all nodes.
        self.root = Node(dom=self, parsed=self.tree.body, parent=None)
        self.nodes = list(self.root.traverse())

    def init_labels(self):
        # Get labeled parsed nodes.
        self.labeled_parsed_nodes = {
            k: self.page.labels.get_labeled_nodes(k)
            for k in self.page.labels.label_keys
        }

        for node in self.nodes:
            node.init_labels()

@dataclasses.dataclass
class Node:
    dom: Dom = dataclasses.field(repr=False)
    parsed: awe.data.parsing.Node
    parent: Optional['Node'] = dataclasses.field(repr=False)
    children: list['Node'] = dataclasses.field(repr=False, default_factory=list)

    label_keys: list[str] = dataclasses.field(default_factory=list)
    """
    Label keys of the node or `[]` if the node doesn't correspond to any target
    attribute.
    """

    def __post_init__(self):
        self.children = list(self._iterate_children())

    @property
    def is_text(self):
        return awe.data.html_utils.is_text(self.parsed)

    @property
    def text(self):
        assert self.is_text
        return self.parsed.text(deep=False)

    @property
    def html_tag(self):
        return self.parsed.tag

    def init_labels(self):
        self.label_keys = [
            k
            for k in self.dom.labeled_parsed_nodes.keys()
            if self.parsed in self.dom.labeled_parsed_nodes[k]
        ]
        for key in self.label_keys:
            self.dom.labeled_nodes.setdefault(key, []).append(self)

    def _iterate_children(self):
        for parsed_node in self.parsed.iter(include_text=True):
            yield Node(dom=self.dom, parsed=parsed_node, parent=self)

    def traverse(self,
        shallow_predicate: Callable[['Node'], bool] = lambda _: True,
        deep_predicate: Callable[['Node'], bool] = lambda _: True,
    ):
        stack = [self]
        while len(stack) != 0:
            node = stack.pop()
            if shallow_predicate(node):
                yield node
            if deep_predicate(node):
                stack.extend(node.children)

    def predict_as(self,
        pred_set: awe.data.graph.pred.PredictionSet,
        label_key: str,
        confidence: float
    ):
        """Marks the node as predicted with the given `label_key`."""

        pred_set.add(label_key, awe.data.graph.pred.NodePrediction(
            node=self,
            confidence=confidence
        ))

    def mark_predicted(self, pred_set: awe.data.graph.pred.PredictionSet):
        pred_set.increment(self)
