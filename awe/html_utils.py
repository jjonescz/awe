import html

import parsel
from lxml import etree
from lxml.html import soupparser

# pylint: disable-next=protected-access
invalid_charrefs_backup = dict(html._invalid_charrefs)

def parse_html(dom: str):
    return parsel.Selector(root=soupparser.fromstring(dom))

def unescape(text: str, with_html_entities: bool = False):
    if with_html_entities:
        # HACK: Process invalid characters as they are. For example, this makes
        # `html.unescape` convert `&#150;` into `\x96` (ASCII) instead of
        # `%u2013` (UTF-8). This is needed for using XPath against HTML that
        # contains such entities, because XPath sees `&#150;` normally as
        # `\x96`.
        if not getattr(html, '_hacked', False):
            # pylint: disable-next=protected-access
            invalid_charrefs = html._invalid_charrefs
            for key in invalid_charrefs:
                invalid_charrefs[key] = chr(key)
            setattr(html, '_hacked', True)
    else:
        # Revert hack.
        if getattr(html, '_hacked', False):
            # pylint: disable-next=protected-access
            html._invalid_charrefs = invalid_charrefs_backup
            setattr(html, '_hacked', False)
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
