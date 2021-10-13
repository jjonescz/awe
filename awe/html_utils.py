import parsel
import html
from lxml import etree

def clean(page: parsel.Selector):
    page.css('script, style').remove()
    page.xpath('//comment()').remove()
    return page

def unescape(text: str):
    # HACK: Process invalid characters as they are, so that it works with XPath.
    if not getattr(html, '_hacked', False):
        for key in html._invalid_charrefs.keys():
            html._invalid_charrefs[key] = chr(key)
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
        parent = root.xpath(f'{node._expr}/..', **kwargs)[0]
        children = parent.xpath('text()')
        # Find child that has the same text as `node`.
        index, _ = next(filter(
            lambda p: p[1].get() == node.get(),
            enumerate(children)
        ))
        return f'{get_xpath(parent)}/text()[{index + 1}]'
    return get_el_xpath(node.root)

def iter_with_fragments(node: etree._Element):
    """
    Gets XPaths of all nodes and text fragments in subtree of `node`.
    """
    for subnode in node.iter():
        subnode_xpath = get_el_xpath(subnode)
        subnode: etree._Element
        yield subnode_xpath, subnode
        for index, text in enumerate(subnode.xpath('text()')):
            text: str
            yield f'{subnode_xpath}/text()[{index + 1}]', text
