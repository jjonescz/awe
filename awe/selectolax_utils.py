import selectolax
import selectolax.parser

TEXT_TAG = '-text'

# pylint: disable-next=c-extension-no-member
def iter_prev(node: selectolax.parser.Node):
    while node.prev is not None:
        yield node.prev
        node = node.prev

# pylint: disable-next=c-extension-no-member
def get_xpath(node: selectolax.parser.Node):
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

# pylint: disable-next=c-extension-no-member
def is_text(node: selectolax.parser.Node):
    return node.tag == TEXT_TAG

# pylint: disable-next=c-extension-no-member
def get_xpath_tag(node: selectolax.parser.Node):
    return 'text()' if is_text(node) else node.tag
