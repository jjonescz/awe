import collections
import dataclasses
from typing import TYPE_CHECKING, Any, Optional
import warnings

import numpy as np
import sklearn.neighbors

import awe.data.html_utils
import awe.data.parsing
import awe.data.visual.structs

if TYPE_CHECKING:
    import awe.data.set.pages


class Dom:
    root: Optional['Node'] = None
    nodes: Optional[list['Node']] = None
    labeled_nodes: dict[str, list['Node']]
    friend_cycles_computed: bool = False
    visual_neighbors_computed: bool = False

    def __init__(self,
        page: 'awe.data.set.pages.Page'
    ):
        self.page = page
        self.labeled_nodes = {}
        self.tree = awe.data.parsing.parse_html(page.get_html_text())

    def init_nodes(self):
        # Get all nodes.
        self.root = Node(dom=self, parsed=self.tree.root, parent=None)
        self.nodes = list(self.root.traverse())
        for idx, node in enumerate(self.nodes):
            node.deep_index = idx

    def filter_nodes(self):
        awe.data.parsing.filter_tree(self.tree)
        self.nodes = [
            node
            for node in self.nodes
            if not node.is_detached
        ]
        for node in self.nodes:
            node.children = [n for n in node.children if not n.is_detached]

    def find_parsed_node(self, node: awe.data.parsing.Node):
        index_path = awe.data.html_utils.get_index_path(node)
        return self.root.find_by_index_path(index_path)

    def init_labels(self, propagate_to_leaves: bool = False):
        # Clear DOM node labeling.
        self.labeled_nodes.clear()
        for node in self.nodes:
            node.label_keys.clear()

        for label_key in self.page.labels.label_keys:
            # Get labeled nodes.
            parsed_nodes = self.page.labels.get_labeled_nodes(label_key)
            if propagate_to_leaves:
                parsed_nodes = awe.data.html_utils.expand_leaves(parsed_nodes)

            # Find the labeled nodes in our DOM.
            labeled_nodes = [self.find_parsed_node(n) for n in parsed_nodes]

            # Fill node labeling.
            self.labeled_nodes[label_key] = labeled_nodes
            for node in labeled_nodes:
                node.label_keys.append(label_key)

    def compute_friend_cycles(self,
        max_ancestor_distance: int = 5,
        max_friends: int = 10,
        only_variable_nodes: bool = True,
    ):
        """Finds friends and partner for each text node (from SimpDOM paper)."""

        descendants = collections.defaultdict(list)

        if only_variable_nodes:
            target_nodes = [n for n in self.nodes if n.is_variable_text]
        else:
            target_nodes = [n for n in self.nodes if n.is_text]

        for node in target_nodes:
            ancestors = node.get_ancestors(max_ancestor_distance)
            for ancestor in ancestors:
                descendants[ancestor].append(node)

        for node in target_nodes:
            ancestors = node.get_ancestors(max_ancestor_distance)
            friends: set[Node] = set()
            for ancestor in ancestors:
                desc = descendants[ancestor]
                if len(desc) == 2:
                    node.partner = [x for x in desc if x != node][0]
                friends.update(desc)

            # Node itself got added to its friends (as its a descendant of its
            # ascendants), but it should not be there.
            if len(ancestors) != 0:
                friends.remove(node)

            # Keep only limited number of closest friends.
            if len(friends) > max_friends:
                closest_friends = sorted(friends,
                    # pylint: disable-next=cell-var-from-loop
                    key=lambda n: n.distance_to(node)
                )
                node.friends = closest_friends[:max_friends]
            else:
                node.friends = list(friends)

            # Keep nodes in DOM order.
            node.friends.sort(key=lambda n: n.deep_index)

        self.friend_cycles_computed = True

    def compute_visual_neighbors(self, n_neighbors: int = 4):
        target_nodes = [
            n for n in self.page.dom.nodes
            if n.is_text and n.box is not None
        ]
        coords = np.array([n.box.center_point for n in target_nodes])
        n_neighbors += 1 # 0th neighbor is the node itself
        nn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors)

        if len(coords) < nn.n_neighbors:
            warnings.warn('Full neighborhood.')
            # Too little samples, everyone is neighbor with everyone else.
            for node in target_nodes:
                neighbors = [
                    VisualNeighbor.create(
                        distance=node.distance_to(n),
                        node=node,
                        neighbor=n
                    )
                    for n in target_nodes
                ]
                neighbors.sort(key=lambda n: n.distance)
                # Repeat the last visual neighbor if there is still too little
                # of them.
                d_neighbors = n_neighbors - len(neighbors)
                if d_neighbors > 0:
                    neighbors.extend([neighbors[-1]] * d_neighbors)
                node.visual_neighbors = neighbors[1:]
            self.visual_neighbors_computed = True
            return

        nn.fit(coords)
        d, i = nn.kneighbors(coords)
        for node, distances, indices in zip(target_nodes, d, i):
            node.visual_neighbors = [
                VisualNeighbor.create(
                    distance=dist,
                    node=node,
                    neighbor=neighbor
                )
                for dist, neighbor in zip(
                    distances[1:],
                    (target_nodes[idx] for idx in indices[1:])
                )
            ]

        self.visual_neighbors_computed = True

    def compute_visual_neighbors_rect(self, n_neighbors: int = 4):
        target_nodes = [
            n for n in self.page.dom.nodes
            if n.is_text and n.box is not None
        ]
        coords = np.array([c for n in target_nodes for c in n.box.corners])
        n_neighbors += 1 # 0th neighbor is the node itself
        nn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors * 4)

        if len(coords) < nn.n_neighbors:
            # Too little samples, cannot compare all corners.
            warnings.warn('Falling back to center neighborhood.')
            self.compute_visual_neighbors(n_neighbors=n_neighbors-1)
            return

        nn.fit(coords)
        d, i = nn.kneighbors(coords)
        for idx, node in enumerate(target_nodes):
            neighbors = [
                VisualNeighbor.create(
                    distance=dist,
                    node=node,
                    neighbor=neighbor
                )
                for distances, indices in zip(
                    d[idx * 4:idx * 4 + 4],
                    i[idx * 4:idx * 4 + 4]
                )
                for dist, neighbor in zip(
                    distances,
                    (target_nodes[idx // 4] for idx in indices)
                )
            ]

            neighbors.sort(key=lambda n: n.distance)

            # Keep only distinct nodes (otherwise, different corners of the same
            # node can be included).
            c = 0
            u = set()
            distinct = []
            for n in neighbors:
                if n.neighbor not in u:
                    u.add(n.neighbor)
                    c += 1
                    distinct.append(n)
                    if c == n_neighbors:
                        break
            node.visual_neighbors = distinct[1:]

        self.visual_neighbors_computed = True

# Setting `eq=False` makes the `Node` inherit hashing and equality functions
# from `Object` (https://stackoverflow.com/a/53990477).
@dataclasses.dataclass(eq=False)
class Node:
    dom: Dom = dataclasses.field(repr=False)
    parsed: awe.data.parsing.Node
    parent: Optional['Node'] = dataclasses.field(repr=False)
    children: list['Node'] = dataclasses.field(repr=False, default_factory=list)

    same_tag_index: Optional[int] = dataclasses.field(repr=False, default=-1)
    """
    Index of this node among its siblings with the same `html_tag`.
    `None` if this is the only such child.
    """

    label_keys: list[str] = dataclasses.field(default_factory=list)
    """
    Label keys of the node or `[]` if the node doesn't correspond to any target
    attribute.
    """

    deep_index: Optional[int] = dataclasses.field(repr=False, default=None)
    """Iteration index of the node inside the `page`."""

    friends: Optional[list['Node']] = dataclasses.field(repr=False, default=None)
    """
    Only set if the current node is a text node. Contains set of text nodes
    where distance to lowest common ancestor with the current node is less than
    or equal to 5. Also limited to 10 closest friends (closest by means of
    `distance_to`).
    """

    partner: Optional['Node'] = dataclasses.field(repr=False, default=None)
    """
    One of `friends` such that the current node and the friend are the only two
    text nodes under a common ancestor. Usually, this is the closest friend.
    """

    is_variable_text: bool = dataclasses.field(repr=False, default=False)
    """Whether this text node is variable across pages in a website."""

    semantic_html_tag: Optional[str] = dataclasses.field(repr=False, default=None)
    """Most semantic HTML tag (found by `HtmlTag` feature)."""

    box: Optional[awe.data.visual.structs.BoundingBox] = \
        dataclasses.field(repr=False, default=None)

    needs_visuals: bool = dataclasses.field(repr=False, default=False)
    """Whether `visuals` should be loaded for this node."""

    visuals: dict[str, Any] = dataclasses.field(init=False, default_factory=dict)
    """`VisualAttribute.name` -> attribute's value or `None`."""

    visual_neighbors: Optional[list['VisualNeighbor']] = \
        dataclasses.field(repr=False, default=None)
    """Closest nodes visually."""

    def __post_init__(self):
        self.children = list(self._iterate_children())

        # Compute `same_tag_index`es.
        tags: dict[str, tuple[int, Node]] = {}
        for child in self.children:
            c, _ = tags.get(child.html_tag, (0, None))
            child.same_tag_index = c
            tags[child.html_tag] = (c + 1, child)
        for c, child in tags.values():
            if c == 1:
                child.same_tag_index = None

    @property
    def id(self):
        return self.parsed.id

    @property
    def is_text(self):
        return awe.data.html_utils.is_text(self.parsed)

    @property
    def text(self):
        assert self.is_text
        return self.parsed.text(deep=False)

    @property
    def is_root(self):
        return self.dom.root == self

    @property
    def is_detached(self):
        return not self.is_root and self.parsed.parent is None

    @property
    def is_empty(self):
        return awe.data.html_utils.is_empty(self.parsed)

    @property
    def html_tag(self):
        return self.parsed.tag

    def get_xpath(self):
        return awe.data.html_utils.get_xpath(self.parsed)

    def get_xpath_element(self):
        tag = self.html_tag if not self.is_text else 'text()'
        if self.same_tag_index is None:
            return tag
        return f'{tag}[{self.same_tag_index + 1}]'

    def get_attributes(self):
        if self.is_text:
            # HACK: Lexbor can crash when querying attributes of text fragments.
            return {}
        return self.parsed.attributes

    def find_by_index_path(self, indices: list[int]):
        """Finds node by output of `awe.data.html_utils.get_index_path`."""
        node = self
        for idx in indices:
            node = node.children[idx]
        return node

    def _iterate_children(self):
        for parsed_node in self.parsed.iter(include_text=True):
            yield Node(dom=self.dom, parsed=parsed_node, parent=self)

    def traverse(self):
        """Iterates tree rooted in the current node in DFS order."""

        stack = [self]
        while len(stack) != 0:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def get_ancestors(self, num: int):
        """
        List of ancestors up to length `num` starting from this node's parent.
        """
        return list(self.iterate_ancestors(num))

    def iterate_ancestors(self, num: int):
        node = self.parent
        for _ in range(num):
            if node is None:
                break
            yield node
            node = node.parent

    def get_ancestor_chain(self, num: int):
        """
        List of ancestors of length exactly `num` (trailing nodes are repeated
        if necessary) starting from the most distant ancestor.

        Most nodes being classified have `num` ancestors and returning a
        fixed-sized list simplifies feature and deep learning computation.
        """
        result = list(self.iterate_ancestor_chain(num))
        result.reverse()
        return result

    def iterate_ancestor_chain(self, num: int):
        node = self
        for _ in range(num):
            if node.parent is not None:
                node = node.parent
            yield node

    def get_all_ancestors(self):
        result = list(self.iterate_all_ancestors())
        result.reverse()
        return result

    def iterate_all_ancestors(self):
        node = self
        while node is not None:
            yield node
            node = node.parent

    def distance_to(self, other: 'Node'):
        return abs(self.deep_index - other.deep_index)

    def get_partner_set(self):
        if self.partner is not None:
            return [self.partner]
        return []

    def unwrap(self, tag_names: set[str]):
        """
        If this node is wrapped in another node from set of `tag_names`, returns
        the parent node (recursively unwrapped).
        """

        node = self
        while node.parent is not None and (
            node.is_text or (
                node.html_tag in tag_names and
                len(node.parent.children) == 1
            )
        ):
            node = node.parent
        return node

@dataclasses.dataclass
class VisualNeighbor:
    distance: float
    distance_x: float
    distance_y: float
    neighbor: Node

    @staticmethod
    def create(distance: float, node: Node, neighbor: Node):
        node_center = node.box.center_point
        neighbor_center = neighbor.box.center_point
        return VisualNeighbor(
            distance=distance,
            distance_x=neighbor_center[0] - node_center[0],
            distance_y=neighbor_center[1] - node_center[1],
            neighbor=neighbor
        )
