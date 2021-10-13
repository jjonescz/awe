import html

import parsel
from lxml import etree


def clean(page: parsel.Selector):
    page.css('script, style').remove()
    page.xpath('//comment()').remove()
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

def get_xpath(
    node: parsel.Selector,
    root: parsel.Selector = None,
    **kwargs
) -> str:
    """Gets absolute XPath for a node."""
    if isinstance(node.root, str):
        # String nodes are complicated.
        # pylint: disable-next=protected-access
        parent = root.xpath(f'{node._expr}/..', **kwargs)[0]
        children = parent.xpath('text()')
        # Find child that has the same text as `node`.
        index, _ = next(filter(
            lambda p: p[1].get() == node.get(),
            enumerate(children)
        ))
        return f'{get_xpath(parent)}/text()[{index + 1}]'
    return get_el_xpath(node.root)

def get_parent_xpath(xpath: str):
    return xpath[:xpath.rindex('/')]
