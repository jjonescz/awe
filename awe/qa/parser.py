import html
import re

import selectolax
import selectolax.parser

from awe import awe_graph, selectolax_utils, utils

WHITESPACE_REGEX = r'(\s|[\u200b])+'
"""Matches whitespace characters."""

def get_page_labels(page: awe_graph.HtmlPage):
    return {
        # Note that labels may contain HTML entities (e.g., `&amp;`), that's
        # why we call `html.unescape` on them.
        f: [
            collapse_whitespace(html.unescape(v))
            for v in page.get_groundtruth_texts(f)
        ]
        for f in page.fields
    }

def get_spans(words: list[str], needle: str):
    return [
        (start, start + len(needle))
        for start in _find_all(words, needle)
    ]

def _find_all(words: list[str], needle: str):
    """Like calling `utils.find_all(' '.join(words), needle)`."""
    start = 0
    for word in words:
        yield from (start + i for i in utils.find_all(word, needle))
        start += len(word) + 1 # add one for space between words

def get_page_words(page: awe_graph.HtmlPage):
    tree = parse_page(page)
    fragments = get_fragments(tree)
    return [text for _, text in fragments]

# pylint: disable-next=c-extension-no-member
def get_fragments(tree: selectolax.parser.HTMLParser):
    return [
        (node, text)
        for node in tree.body.traverse(include_text=True)
        if (
            selectolax_utils.is_text(node) and
            not is_empty_or_space(text := get_node_text(node))
        )
    ]

def is_empty_or_space(text: str):
    return text == '' or text.isspace()

# pylint: disable-next=c-extension-no-member
def get_node_text(node: selectolax.parser.Node):
    return collapse_whitespace(node.text())

def parse_page(page: awe_graph.HtmlPage):
    # pylint: disable-next=c-extension-no-member
    tree = selectolax.parser.HTMLParser(page.html)

    # Ignore some tags.
    tree.strip_tags([
        'script',
        'style',
        'head',
        '[document]',
        'noscript',
        'iframe'
    ])

    return tree

def collapse_whitespace(text: str):
    return re.sub(WHITESPACE_REGEX, ' ', text)
