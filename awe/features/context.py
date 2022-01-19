import collections
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from awe import filtering

if TYPE_CHECKING:
    from awe import awe_graph

@dataclass
class CategoryInfo:
    unique_id: int
    """Unique ID of the category."""

    count: int
    """How many times was this category used."""

    def merge_with(self, other: 'CategoryInfo'):
        self.count += other.count

def categorical_dict() -> dict[str, CategoryInfo]:
    d = collections.defaultdict()
    # Note that ID `0` is reserved for "unseen" category.
    d.default_factory = lambda: CategoryInfo(len(d) + 1, 0)
    return d

@dataclass
class AweFeatureOptions:
    cutoff_words: Optional[int] = 15
    """
    Maximum number of words to preserve in each node (or `None` to preserve
    all). Used by `CharacterIdentifiers` and `WordIdentifiers`.
    """

    cutoff_word_length: Optional[int] = 10
    """
    Maximum number of characters to preserve in each token (or `None` to
    preserve all). Used by `CharacterIdentifiers`.
    """

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

    options: AweFeatureOptions = AweFeatureOptions()

    visual_categorical: collections.defaultdict[str,
        collections.defaultdict[str, CategoryInfo]]
    """
    Visual categorical features (name of feature -> category label -> category
    info). Used by `Visuals`.
    """

    def __init__(self):
        self.pages = set()
        self.chars = set()
        self.visual_categorical = collections.defaultdict(categorical_dict)

    def options_from(self, other: 'RootContext'):
        self.options = dataclasses.replace(other.options) # clone

    def merge_with(self, other: 'RootContext'):
        # Check that options are consistent.
        assert self.options == other.options, \
            f'Options {self.options} do not match {other.options}'

        self.pages.update(other.pages)
        self.chars.update(other.chars)
        self.max_word_len = max(self.max_word_len, other.max_word_len)
        self.max_num_words = max(self.max_num_words, other.max_num_words)

        # Merge `visual_categorical` dictionaries.
        for f, other_category in other.visual_categorical.items():
            curr_category = self.visual_categorical.get(f)
            if curr_category is None:
                self.visual_categorical[f] = other_category
            else:
                for l, other_info in other_category.items():
                    curr_info = curr_category.get(l)
                    if curr_info is None:
                        curr_category[l] = other_info
                    else:
                        curr_info.merge_with(other_info)

    def describe_visual_categorical(self):
        return {
            # Place most used first.
            feature: dict(sorted(
                category.items(),
                key=lambda p: p[1].count,
                reverse=True
            ))
            for feature, category in self.visual_categorical.items()
        }

    def total_visual_categorical_count(self):
        return sum(
            i.count
            for c in self.visual_categorical.values()
            for i in c.values()
        )

    def describe(self):
        return {
            'pages': len(self.pages),
            'chars': len(self.chars),
            'max_num_words': self.max_num_words,
            'max_word_len': self.max_word_len,
            'visual_categorical': self.total_visual_categorical_count()
        }

    def freeze(self):
        # Needed to pickle this object.
        for value in self.visual_categorical.values():
            value.default_factory = None
        self.visual_categorical.default_factory = None

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

class PageContextBase:
    """Everything needed to prepare `HtmlPage`."""

    _nodes: list['awe_graph.HtmlNode'] = None

    def __init__(self,
        page: 'awe_graph.HtmlPage',
        node_predicate: filtering.NodePredicate
    ):
        self.page = page
        self.node_predicate = node_predicate

    @property
    def nodes(self):
        """Cached list of `page.nodes`."""
        if self._nodes is None:
            root = self.page.get_tree()
            self._nodes = list(root.iterate_descendants(
                self.node_predicate.include_node_itself,
                self.node_predicate.include_node_descendants
            ))
        return self._nodes

    def prepare(self):
        # Assign indices to nodes (different from `HtmlNode.index` as that
        # one is from before filtering). This is needed to compute edges.
        for index, node in enumerate(self.nodes):
            node.dataset_index = index

class PageContext(PageContextBase):
    """
    Everything needed to compute a `HtmlNode`'s `Feature`s.

    Data stored here are scoped to one `HtmlPage`.
    """

    max_depth: Optional[int] = None
    """Maximum DOM tree depth. Stored by `Depth`."""

    def __init__(self,
        live: LiveContext,
        page: 'awe_graph.HtmlPage',
        node_predicate: filtering.NodePredicate
    ):
        self.live = live
        super().__init__(page, node_predicate)

    @property
    def root(self):
        return self.live.root
