import re

import selectolax
import selectolax.lexbor

import awe.data.html_utils

WHITESPACE_REGEX = r'(\s|[\u200b])+'
"""Matches whitespace characters."""

# pylint: disable=c-extension-no-member
Node = selectolax.lexbor.LexborNode
Tree = selectolax.lexbor.LexborHTMLParser
Error = selectolax.lexbor.SelectolaxError
# pylint: enable=c-extension-no-member

def find_nodes_with_text(tree: Tree, needle: str):
    """
    Finds nodes containing the specified `needle` as their text content.
    """

    normalized_needle = normalize_node_text(needle)
    return [
        node for node in tree.body.traverse(include_text=True)
        if node_contains_normalized_text(node, normalized_needle)
    ]

def node_contains_text(node: Node, needle: str):
    normalized_needle = normalize_node_text(needle)
    return node_contains_normalized_text(node, normalized_needle)

def node_contains_normalized_text(node: Node, normalized_needle: str):
    return (
        awe.data.html_utils.is_text(node) and
        normalize_node_text(node.text()) == normalized_needle
    )

def parse_html(html_text: str):
    # Note that unlike the default selectolax parser, the Lexbor parser can
    # correctly extract text fragments `X`, `Y`, `Z` from HTML
    # `<p>X<br>Y<br>Z</p>`.

    tree = Tree(html_text)

    # Ignore some tags.
    tree.strip_tags([
        'script',
        'style',
        'head',
        'noscript',
        'iframe'
    ])

    # Ignore more tags.
    to_destroy = [n
        for n in tree.root.traverse(include_text=False)
        if awe.data.html_utils.is_comment(n)
    ]
    for n in to_destroy:
        n: Node
        n.decompose(recursive=False)

    return tree

def normalize_node_text(text: str):
    return collapse_whitespace(text).strip()

def collapse_whitespace(text: str):
    return re.sub(WHITESPACE_REGEX, ' ', text)

def remove_whitespace(text: str):
    return re.sub(WHITESPACE_REGEX, '', text)
