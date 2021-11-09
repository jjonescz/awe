import html

import parsel
from lxml import etree
from lxml.html import soupparser


def parse_html(html: str):
    return parsel.Selector(root=soupparser.fromstring(html))

def unescape(text: str):
    # HACK: Process invalid characters as they are, so that it works with XPath.
    if not getattr(html, '_hacked', False):
        # pylint: disable-next=protected-access
        invalid_charrefs = html._invalid_charrefs
        for key in invalid_charrefs:
            invalid_charrefs[key] = chr(key)
        setattr(html, '_hacked', True)
    return html.unescape(text)

def get_el_xpath(node: etree._Element) -> str:
    return node.getroottree().getpath(node)

def find_fragment_index(parents: parsel.SelectorList, text_fragment: str):
    for parent in parents:
        parent: parsel.Selector
        children = list(enumerate(parent.xpath('text()')))
        for index, fragment in children:
            fragment: parsel.Selector
            if fragment.get() == text_fragment:
                return index, parent, len(children)
    raise LookupError()

def get_xpath(
    node: parsel.Selector,
    root: parsel.Selector = None,
    **kwargs
) -> str:
    """Gets absolute XPath for a node."""
    if isinstance(node.root, str):
        # String nodes are complicated.
        # pylint: disable-next=protected-access
        parents = root.xpath(f'{node._expr}/..', **kwargs)
        index, parent, count = find_fragment_index(parents, node.get())
        xpath = f'{get_xpath(parent)}/text()'
        if count > 1:
            # Append index only if there are multiple text nodes.
            xpath += f'[{index + 1}]'
        return xpath
    return get_el_xpath(node.root)

def get_parent_xpath(xpath: str):
    return xpath[:xpath.rindex('/')]
