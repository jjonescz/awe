import parsel
import html

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

def get_xpath(node: parsel.Selector, root: parsel.Selector = None, **kwargs):
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
    return node.root.getroottree().getpath(node.root)
