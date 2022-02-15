from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import awe.data.parsing

TEXT_TAG = '-text'

def iter_prev(node: 'awe.data.parsing.Node'):
    while node.prev is not None:
        yield node.prev
        node = node.prev

def get_xpath(node: 'awe.data.parsing.Node'):
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

def is_text(node: 'awe.data.parsing.Node'):
    return node.tag == TEXT_TAG

def get_xpath_tag(node: 'awe.data.parsing.Node'):
    return 'text()' if is_text(node) else node.tag
