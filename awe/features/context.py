from typing import TYPE_CHECKING, Optional

from awe import filtering

if TYPE_CHECKING:
    from awe import awe_graph


class RootContext:
    """
    Data stored here are scoped to all pages. Initialized in `Feature.prepare`.
    """

    pages: set[str]
    """Identifiers of pages used for feature preparation against this object."""

    chars: set[str]
    """
    All characters present in processed nodes. Stored by `CharacterIdentifiers`.
    """

    max_word_len: int = 0
    """Length of the longest word. Stored by `CharacterIdentifiers`."""

    max_num_words: int = 0
    """
    Number of words in the longest node (up to `cutoff_words`). Stored by
    `CharacterIdentifiers` and `WordIdentifiers`.
    """

    cutoff_words: Optional[int] = None
    """
    Maximum number of words to preserve in each node (or `None` to preserve
    all). Used by `CharacterIdentifiers` and `WordIdentifiers`.
    """

    def __init__(self):
        self.pages = set()
        self.chars = set()

    def options_from(self, other: 'RootContext'):
        self.cutoff_words = other.cutoff_words

    def merge_with(self, other: 'RootContext'):
        self.pages.update(other.pages)
        self.chars.update(other.chars)
        self.max_word_len = max(self.max_word_len, other.max_word_len)
        self.max_num_words = max(self.max_num_words, other.max_num_words)
        assert self.cutoff_words == other.cutoff_words, \
            f'Option `cutoff_words` does not match ({self.cutoff_words} ' + \
            'vs. {other.cutoff_words})'

    def extract_options(self):
        return {
            'cutoff_words': self.cutoff_words
        }

class LiveContext:
    """
    Non-persisted (live) data scoped to all pages. Initialized in
    `Feature.initialize`.
    """

    char_dict: dict[str, int]
    """Used by `CharacterEmbedding`."""

    token_dict: dict[str, int]
    """Used by `WordEmbedding`."""

    def __init__(self, root: RootContext):
        self.root = root
        self.char_dict = {}
        self.word_dict = {}

class PageContext:
    """
    Everything needed to compute a `HtmlNode`'s `Feature`s.

    Data stored here are scoped to one `HtmlPage`.
    """

    max_depth: Optional[int] = None
    """Maximum DOM tree depth. Stored by `Depth`."""

    _nodes: list['awe_graph.HtmlNode'] = None

    def __init__(self,
        live: LiveContext,
        page: 'awe_graph.HtmlPage',
        node_predicate: filtering.NodePredicate
    ):
        self.live = live
        self.page = page
        self.node_predicate = node_predicate

    @property
    def root(self):
        return self.live.root

    @property
    def nodes(self):
        """Cached list of `page.nodes`."""
        if self._nodes is None:
            root = self.page.initialize_tree()
            self._nodes = list(root.iterate_descendants(
                self.node_predicate.include_node))
        return self._nodes
