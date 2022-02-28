import collections
import dataclasses
from typing import TYPE_CHECKING, Callable, Optional

import awe.data.html_utils
import awe.data.parsing

if TYPE_CHECKING:
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
        for idx, node in enumerate(self.nodes):
            node.deep_index = idx

    def init_labels(self):
        # Get labeled parsed nodes.
        self.labeled_parsed_nodes = {
            k: self.page.labels.get_labeled_nodes(k)
            for k in self.page.labels.label_keys
        }

        for node in self.nodes:
            node.init_labels()

    def compute_friend_cycles(self,
        max_ancestor_distance: int = 5,
        max_friends: int = 10,
    ):
        """Finds friends and partner for each node (from SimpDOM paper)."""

        descendants = collections.defaultdict(list)

        for node in self.nodes:
            ancestors = node.get_ancestors(max_distance=max_ancestor_distance)
            for ancestor in ancestors:
                descendants[ancestor].append(node)

        for node in self.nodes:
            ancestors = node.get_ancestors(max_distance=max_ancestor_distance)
            for ancestor in ancestors:
                desc = descendants[ancestor]
                if len(desc) == 2:
                    node.partner = [x for x in desc if x != node][0]
                node.friends.update(desc)

            # Node itself got added to its friends (as its a descendant of its
            # ascendants), but it should not be there.
            if len(ancestors) != 0:
                node.friends.remove(node)

            # Keep only limited number of closest friends.
            if len(node.friends) > max_friends:
                closest_friends = sorted(node.friends,
                    # pylint: disable-next=cell-var-from-loop
                    key=lambda n: n.distance_to(node),
                    reverse=True
                )
                node.friends = set(closest_friends[:max_friends])

# Setting `eq=False` makes the `Node` inherit hashing and equality functions
# from `Object` (https://stackoverflow.com/a/53990477).
@dataclasses.dataclass(eq=False)
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

    deep_index: Optional[int] = dataclasses.field(repr=False, default=None)
    """Iteration index of the node inside the `page`."""

    partner: Optional['Node'] = dataclasses.field(repr=False, default=None)
    friends: set['Node'] = dataclasses.field(repr=False, default_factory=set)

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

    def get_ancestors(self, max_distance: int):
        if self.parent is None or max_distance <= 0:
            return []
        return [self.parent] + self.parent.get_ancestors(max_distance - 1)

    def distance_to(self, other: 'Node'):
        return abs(self.deep_index - other.deep_index)
