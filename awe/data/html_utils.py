"""Utilities for working with DOM trees."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import awe.data.parsing

TEXT_TAG = '-text'
COMMENT_TAG = '-comment'

def iter_prev(node: 'awe.data.parsing.Node'):
    """Iterate previous siblings of `node`."""

    while node.prev is not None:
        yield node.prev
        node = node.prev

def get_xpath(node: 'awe.data.parsing.Node'):
    """Construct absolute XPath uniquely identifying `node`."""

    xpath = ''
    while node.parent is not None:
        # How many nodes of the same tag are there?
        count = sum(1
            for sibling in node.parent.iter(include_text=True)
            if sibling.tag == node.tag
        )

        # If more than one, attach indexer to XPath.
        indexer = ''
        if count > 1:
            # Determine index of this node inside parent.
            prev_count = sum(1
                for prev in iter_prev(node)
                if prev.tag == node.tag
            )
            indexer = f'[{prev_count + 1}]'

        xpath = f'/{get_xpath_tag(node)}{indexer}{xpath}'

        node = node.parent
    return xpath

def get_index_path(node: 'awe.data.parsing.Node'):
    """
    Gets path of indices that can be used to get to the same node in an
    analogous DOM tree.

    This is similar to `get_xpath` but the resulting index path is more
    condensed as it only consists of numbers (one per level) and does not
    contain any HTML tag names.
    """

    indices: list[int] = []
    while node.parent.parent is not None:
        # Determine index of this node inside parent.
        prev_count = sum(1 for _ in iter_prev(node))
        indices.append(prev_count)
        node = node.parent
    indices.reverse()
    return tuple(indices)

def is_text(node: 'awe.data.parsing.Node'):
    """Determines whether `node` is a text fragment."""

    return node.tag == TEXT_TAG

def is_comment(node: 'awe.data.parsing.Node'):
    """Determines whether `node` represents an HTML comment."""

    return node.tag == COMMENT_TAG

def is_leaf(node: 'awe.data.parsing.Node'):
    """
    Determines whether `node` does not have any children (except text fragments).
    """

    return node.child is None

def is_empty(node: 'awe.data.parsing.Node'):
    """Determines whether `node` is leaf and does not have any text content."""

    return node.child is None and not node.text(deep=False)

def get_xpath_tag(node: 'awe.data.parsing.Node'):
    """Gets HTML tag that can be used in XPath."""

    return 'text()' if is_text(node) else node.tag

def expand_leaves(nodes: list['awe.data.parsing.Node']):
    """Inner nodes are expanded to all their leaf descendants."""

    return [list(iter_leaves(n)) for n in nodes]

def iter_leaves(node: 'awe.data.parsing.Node'):
    """Iterates leaf descendants of `node`."""

    if is_leaf(node):
        yield node
    else:
        for n in node.traverse(include_text=True):
            n: 'awe.data.parsing.Node'
            if is_leaf(n):
                yield n

def expand_descendants(
    nodes: list['awe.data.parsing.Node']
) -> list[list['awe.data.parsing.Node']]:
    """Inner nodes are expanded to all their descendants."""

    return [list(n.traverse(include_text=True)) for n in nodes]
