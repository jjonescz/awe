import re

import selectolax
import selectolax.lexbor

import awe.selectolax_utils

WHITESPACE_REGEX = r'(\s|[\u200b])+'
"""Matches whitespace characters."""

# pylint: disable-next=c-extension-no-member
def find_nodes_with_text(tree: selectolax.lexbor.LexborHTMLParser, needle: str):
    """
    Finds nodes containing the specified `needle` as their text content.
    """

    needle = normalize_node_text(needle)
    return [
        node for node in tree.body.traverse(include_text=True)
        if (
            awe.selectolax_utils.is_text(node) and
            normalize_node_text(node.text()) == needle
        )
    ]

def parse_html(html_text: str):
    # Note that unlike the default selectolax parser, the Lexbor parser can
    # correctly extract text fragments `X`, `Y`, `Z` from HTML
    # `<p>X<br>Y<br>Z</p>`.

    # pylint: disable-next=c-extension-no-member
    tree = selectolax.lexbor.LexborHTMLParser(html_text)

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

def normalize_node_text(text: str):
    return collapse_whitespace(text).strip()

def collapse_whitespace(text: str):
    return re.sub(WHITESPACE_REGEX, ' ', text)

def remove_whitespace(text: str):
    return re.sub(WHITESPACE_REGEX, '', text)
