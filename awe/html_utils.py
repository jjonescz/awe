import html

import parsel
from lxml import etree


def clean(page: parsel.Selector):
    page.css('script, style').remove()
    # Note that root elements cannot be removed, hence the `/*` prefix.
    page.xpath('/*//comment()').remove()
    return page

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
        for index, fragment in enumerate(parent.xpath('text()')):
            fragment: parsel.Selector
            if fragment.get() == text_fragment:
                return index, parent
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
        index, parent = find_fragment_index(parents, node.get())
        return f'{get_xpath(parent)}/text()[{index + 1}]'
    return get_el_xpath(node.root)

def get_parent_xpath(xpath: str):
    return xpath[:xpath.rindex('/')]
