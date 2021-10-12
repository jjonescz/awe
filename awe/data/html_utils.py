import parsel
import html

def clean(page: parsel.Selector):
    page.css('script, style').remove()
    return page

def unescape(text: str):
    # HACK: Process invalid characters as they are, so that it works with XPath.
    for key in html._invalid_charrefs.keys():
        html._invalid_charrefs[key] = chr(key)
    return html.unescape(text)
