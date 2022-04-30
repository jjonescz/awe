"""DOM graph."""

import collections
import dataclasses
import math
import warnings
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import sklearn.neighbors

import awe.data.html_utils
import awe.data.parsing
import awe.data.visual.structs

if TYPE_CHECKING:
    import awe.data.set.pages


class Dom:
    """Document Object Model of a page parsed from its HTML."""

    root: Optional['Node'] = None
    """The root `<html>` node."""

    nodes: Optional[list['Node']] = None
    """All nodes in the graph."""

    labeled_nodes: dict[str, list[list['Node']]]
    """
    Target node groups for each attribute key.

    See also `Page.labeled_nodes`.
    """

    friend_cycles_computed: bool = False
    """A flag whether `compute_friend_cycles` method has been called."""

    visual_neighbors_computed: bool = False
    """A flag whether `compute_visual_neighbors` method has been called."""

    def __init__(self,
        page: 'awe.data.set.pages.Page'
    ):
        self.page = page
        self.labeled_nodes = {}
        self.tree = awe.data.parsing.parse_html(page.get_html_text())

    def init_nodes(self, filter_tree: bool = False):
        """
        Creates `Node` high-level wrappers around parsed nodes.

        Parameters:
        - `filter_tree`: perform equivalent of `filter_nodes` immediately.
        """

        if filter_tree:
            awe.data.parsing.filter_tree(self.tree)

        # Initialize high-level object tree wrapping low-level parsed nodes.
        # Note that the traversal is performed in a strict DFS order.
        deep_index = 0
        root = Node(dom=self, parsed=self.tree.root, parent=None)
        nodes = []
        stack = [root]
        while len(stack) != 0:
            node = stack.pop()
            nodes.append(node)
            node.deep_index = deep_index
            node.create_children()
            stack.extend(reversed(node.children))
            deep_index += 1

        self.root = root
        self.nodes = nodes

    def filter_nodes(self):
        """
        Filters white-space text fragments.

        See also `awe.data.parsing.filter_nodes`.
        """

        awe.data.parsing.filter_tree(self.tree)
        self.nodes = [
            node
            for node in self.nodes
            if not node.is_detached
        ]
        for node in self.nodes:
            node.children = [n for n in node.children if not n.is_detached]

    def find_parsed_node(self, node: awe.data.parsing.Node):
        """
        Finds high-level `Node` wrapper corresponding to low-level parsed node.
        """

        index_path = awe.data.html_utils.get_index_path(node)
        return self.root.find_by_index_path(index_path)

    def init_labels(self,
        propagate_to_leaves: bool = False,
        propagate_to_descendants: bool = False,
    ):
        """
        Populates `labeled_nodes` and `Node.label_keys` of each node.

        Parameters:

        - `propagate_to_leaves`: propagates labels from inner nodes to their
          leaf descendants,
        - `propagate_to_descendants`: propagates labels from inner nodes to all
          their descendants.
        """

        # Clear DOM node labeling.
        self.labeled_nodes.clear()
        for node in self.nodes:
            node.label_keys.clear()

        for label_key in self.page.labels.label_keys:
            # Get labeled nodes.
            parsed_nodes = self.page.labels.get_labeled_nodes(label_key)
            if propagate_to_descendants:
                parsed_nodes = awe.data.html_utils.expand_descendants(parsed_nodes)
            elif propagate_to_leaves:
                parsed_nodes = awe.data.html_utils.expand_leaves(parsed_nodes)
            else:
                parsed_nodes = [parsed_nodes]

            # Find the labeled nodes in our DOM.
            labeled_nodes = [
                [self.find_parsed_node(n) for n in group]
                for group in parsed_nodes
            ]

            # Fill node labeling.
            self.labeled_nodes[label_key] = labeled_nodes
            for group_idx, group in enumerate(labeled_nodes):
                for node in group:
                    node.label_keys.append((label_key, group_idx))

        # Save labeled node identities (preserved when DOM is dropped).
        self.page.labeled_nodes = {
            key: [
                [node.get_identity() for node in group]
                for group in groups
            ]
            for key, groups in self.labeled_nodes.items()
        }

    def compute_friend_cycles(self,
        max_ancestor_distance: int = 5,
        max_friends: int = 10,
    ):
        """Finds friends and partner for each text node (from SimpDOM paper)."""

        descendants = collections.defaultdict(list)

        sample_nodes = [n for n in self.nodes if n.sample]

        for node in sample_nodes:
            ancestors = node.get_ancestors(max_ancestor_distance)
            for ancestor in ancestors:
                descendants[ancestor].append(node)

        for node in sample_nodes:
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
        """
        Finds `n_neighbors` closest `Node.visual_neighbors` for each node.

        Uses distance of node centers to determine closest nodes.
        """

        sample_nodes = [n for n in self.nodes if n.sample]
        coords = np.array([n.box.center_point for n in sample_nodes])
        n_neighbors += 1 # 0th neighbor is the node itself
        nn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors)

        if len(coords) < nn.n_neighbors:
            warnings.warn(f'Full neighborhood ({self.page.html_path!r}).')
            # Too little samples, everyone is neighbor with everyone else.
            for node in sample_nodes:
                neighbors = [
                    VisualNeighbor.create(
                        distance=node.distance_to(n),
                        node=node,
                        neighbor=n
                    )
                    for n in sample_nodes
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
        for node, distances, indices in zip(sample_nodes, d, i):
            node.visual_neighbors = [
                VisualNeighbor.create(
                    distance=dist,
                    node=node,
                    neighbor=neighbor
                )
                for dist, neighbor in zip(
                    distances[1:],
                    (sample_nodes[idx] for idx in indices[1:])
                )
            ]

        self.visual_neighbors_computed = True

    def compute_visual_neighbors_rect(self, n_neighbors: int = 4):
        """
        Finds `n_neighbors` closest `Node.visual_neighbors` for each node.

        Uses distance between all node corners to determine closest nodes.
        """

        sample_nodes = [n for n in self.nodes if n.sample]
        coords = np.array([c for n in sample_nodes for c in n.box.corners])
        n_neighbors += 1 # 0th neighbor is the node itself
        nn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors * 4)

        if len(coords) < nn.n_neighbors:
            # Too little samples, cannot compare all corners.
            warnings.warn('Falling back to center neighborhood ' +
                f'({self.page.html_path!r}).')
            self.compute_visual_neighbors(n_neighbors=n_neighbors-1)
            return

        nn.fit(coords)
        d, i = nn.kneighbors(coords)
        for idx, node in enumerate(sample_nodes):
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
                    (sample_nodes[idx // 4] for idx in indices)
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
    """High-level wrapper around parsed node."""

    dom: Dom = dataclasses.field(repr=False)
    """The parent DOM tree."""

    parsed: awe.data.parsing.Node
    """The wrapped low-level parsed node."""

    parent: Optional['Node'] = dataclasses.field(repr=False)
    """Immediate ancestor of the node in the DOM tree (`None` for root node)."""

    children: list['Node'] = dataclasses.field(repr=False, default_factory=list)
    """Immediate descendants of the node in the DOM tree."""

    same_tag_index: Optional[int] = dataclasses.field(repr=False, default=-1)
    """
    Index of this node among its siblings with the same `html_tag`.
    `None` if this is the only such child.
    """

    label_keys: list[(str, int)] = dataclasses.field(default_factory=list)
    """
    Label keys and group indices of the node or `[]` if the node doesn't
    correspond to any target attribute.
    """

    deep_index: Optional[int] = dataclasses.field(repr=False, default=None)
    """Iteration index of the node inside the `page`."""

    sample: bool = dataclasses.field(default=False)
    """Whether this node has been selected for classification."""

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

    semantic_html_tag: Optional[str] = dataclasses.field(repr=False, default=None)
    """Most semantic HTML tag (found by `HtmlTag` feature)."""

    box: Optional[awe.data.visual.structs.BoundingBox] = \
        dataclasses.field(repr=False, default=None)
    """Visual coordinates of the node when rendered on a page."""

    needs_visuals: bool = dataclasses.field(repr=False, default=False)
    """Whether `visuals` should be loaded for this node."""

    visuals: dict[str, Any] = dataclasses.field(init=False, default_factory=dict)
    """`VisualAttribute.name` -> attribute's value or `None`."""

    visual_neighbors: Optional[list['VisualNeighbor']] = \
        dataclasses.field(repr=False, default=None)
    """Closest nodes visually."""

    @property
    def id(self):
        """Value of the `id` HTML attribute."""

        return self.parsed.id

    @property
    def is_text(self):
        """Whether this node is a text fragment."""

        return awe.data.html_utils.is_text(self.parsed)

    @property
    def text(self):
        """Text content of the text fragment."""

        assert self.is_text
        return self.parsed.text(deep=False)

    @property
    def is_root(self):
        """Whether this is the root `<html>` node."""

        return self.dom.root == self

    @property
    def is_detached(self):
        """Whether this node is still part of the `dom` tree."""

        return not self.is_root and self.parsed.parent is None

    @property
    def is_empty(self):
        """Whether this node is leaf and does not have any text content."""

        return awe.data.html_utils.is_empty(self.parsed)

    @property
    def is_leaf(self):
        """Whether `node` does not have any children (except text fragments)."""

        return awe.data.html_utils.is_leaf(self.parsed)

    @property
    def html_tag(self):
        """HTML tag name of this node."""

        return self.parsed.tag

    def get_text_or_tag(self):
        """Representation of this node for displaying."""

        if self.is_text:
            return f'={self.text}'
        return f'<{self.html_tag}>'

    def get_attribute(self, name: str, default = None):
        """Retrieves HTML attribute `name` or returns `default` value."""

        return self.get_attributes().get(name, default)

    def get_xpath(self):
        """Constructs absolute XPath of this node."""

        return awe.data.html_utils.get_xpath(self.parsed)

    def get_xpath_element(self):
        """Constructs XPath element (tag name + indexer) of this node."""

        tag = self.html_tag if not self.is_text else 'text()'
        if self.same_tag_index is None:
            return tag
        return f'{tag}[{self.same_tag_index + 1}]'

    def get_attributes(self):
        """Retrieves all HTML attributes of this node as dictionary."""

        if self.is_text:
            # HACK: Lexbor can crash when querying attributes of text fragments.
            return {}
        return self.parsed.attributes

    def get_identity(self):
        """Constructs `NodeIdentity` for this node."""

        return NodeIdentity(
            page=self.dom.page,
            index_path = awe.data.html_utils.get_index_path(self.parsed),
        )

    def find_by_index_path(self, indices: list[int]):
        """Finds node by output of `awe.data.html_utils.get_index_path`."""

        node = self
        for idx in indices:
            node = node.children[idx]
        return node

    def create_children(self):
        """Initializes the `children` with `Node` instances."""

        self.children = [
            Node(dom=self.dom, parsed=parsed_node, parent=self)
            for parsed_node in self.parsed.iter(include_text=True)
        ]

        # Compute `same_tag_index`es.
        tags: dict[str, tuple[int, Node]] = {}
        for child in self.children:
            c, _ = tags.get(child.html_tag, (0, None))
            child.same_tag_index = c
            tags[child.html_tag] = (c + 1, child)
        for c, child in tags.values():
            if c == 1:
                child.same_tag_index = None

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
        """Lazy version of `get_ancestors`."""

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
        """Lazy version of `get_ancestor_chain`."""

        node = self
        for _ in range(num):
            if node.parent is not None:
                node = node.parent
            yield node

    def get_all_ancestors(self):
        """
        Like `get_ancestor_chain` but obtains all ancestors up to the root.
        """

        result = list(self.iterate_all_ancestors())
        result.reverse()
        return result

    def iterate_all_ancestors(self):
        """Lazy version of `get_all_ancestors`."""

        node = self
        while node is not None:
            yield node
            node = node.parent

    def distance_to(self, other: 'Node'):
        """Computes distance to `other` node in the DOM tree."""

        return abs(self.deep_index - other.deep_index)

    def get_partner_set(self):
        """Set of `partner` nodes."""

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
                (node is self or len(node.parent.children) == 1)
            )
        ):
            node = node.parent
        return node

    def find_semantic_ancestor(self):
        """Finds most semantic ancestor of the node."""

        return self.unwrap(tag_names={ 'span', 'div' })

    def find_semantic_html_tag(self):
        """Finds most semantic HTML tag for the node."""

        semantic = self.find_semantic_ancestor()
        return semantic.html_tag

@dataclasses.dataclass(frozen=True)
class NodeIdentity:
    """
    Unique identity of a node inside its DOM tree.

    This can be held separately from `Node` instance, hence releasing memory for
    DOM trees, while still being able to identify some (e.g., target) nodes.
    """

    page: 'awe.data.set.pages.Page'
    """Page where the node can be found."""

    index_path: tuple[int]
    """Output of `awe.data.html_utils.get_index_path`."""

    def find_node(self):
        """
        Finds `Node` instance corresponding to this identity in `page`'s DOM
        tree.
        """

        return self.page.dom.root.find_by_index_path(self.index_path)

@dataclasses.dataclass
class VisualNeighbor:
    """
    Represents one of N closest visual neighbors of a `Node`.

    Encapsulates output of `Dom.compute_visual_neighbors`.
    """

    distance: float
    """
    Distance used to find this visual neighbor as one of the N closest ones.

    This depends on the method called to obtain this instance (more precisely,
    whether or not it has the `_rect` suffix).
    """

    distance_x: float
    """Horizontal distance between node's and its neighbor's centers."""

    distance_y: float
    """Vertical distance between node's and its neighbor's centers."""

    neighbor: Node
    """The neighbor `Node`."""

    @staticmethod
    def create(distance: float, node: Node, neighbor: Node):
        """Creates new visual `neighbor` of `node`."""

        node_center = node.box.center_point
        neighbor_center = neighbor.box.center_point
        return VisualNeighbor(
            distance=distance,
            distance_x=neighbor_center[0] - node_center[0],
            distance_y=neighbor_center[1] - node_center[1],
            neighbor=neighbor
        )

    def get_visual_distance(self, normalize: bool):
        """Computes visual distance feature vector."""

        if not normalize:
            return (self.distance_x, self.distance_y, self.distance)

        root_box = self.neighbor.dom.root.box
        return (
            _safe_log(self.distance_x / root_box.width),
            _safe_log(self.distance_y / root_box.width),
            _safe_log(self.distance / root_box.width)
        )

def _safe_log(x: float):
    if x > 0:
        return math.log(x)
    return 0
