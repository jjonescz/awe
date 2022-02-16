import dataclasses
from typing import Callable, Optional

import awe.data.parsing
import awe.data.set.pages


class Dom:
    def __init__(self, page: awe.data.set.pages.Page):
        self.page = page
        self.tree = awe.data.parsing.parse_html(page.get_html_text())

        # Get all nodes.
        self.root = Node(dom=self, parsed=self.tree.body, parent=None)
        self.nodes = list(self.root.traverse())

@dataclasses.dataclass
class Node:
    dom: Dom = dataclasses.field(repr=False)
    parsed: awe.data.parsing.Node
    parent: Optional['Node'] = dataclasses.field(repr=False)
    children: list['Node'] = dataclasses.field(repr=False, default_factory=list)

    def __post_init__(self):
        self.children = list(self._iterate_children())

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
