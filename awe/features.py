import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, TypeVar

import torch
from torchtext.data import utils as text_utils

from awe import filtering

if TYPE_CHECKING:
    from awe import awe_graph

T = TypeVar('T', bound='Feature') # pylint: disable=invalid-name

class RootContext:
    """
    Data stored here are scoped to all pages. Initialized in `Feature.prepare`.
    """

    pages: set[str]
    """Identifiers of pages used for feature preparation against this object."""

    chars: set[str]
    """
    All characters present in processed nodes. Stored by `CharacterEmbedding`.
    """

    tokens: set[str]
    """
    All word tokens present in processed nodes. Stored by `WordEmbedding`.
    """

    def __init__(self):
        self.pages = set()
        self.chars = set()
        self.tokens = set()

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

class Feature(ABC):
    @property
    @abstractmethod
    def summary(self):
        """User-friendly details."""

    def initialize(self, context: LiveContext):
        """Work needed to be done so that this feature can be computed."""

    def prepare(self, node: 'awe_graph.HtmlNode', context: RootContext):
        """
        Prepares feature for the given `node`.

        This method runs for all nodes before initializing and computing the
        features. Can be used for example to populate a global word dictionary.
        """

    @abstractmethod
    def compute(self,
        node: 'awe_graph.HtmlNode',
        context: PageContext) -> torch.FloatTensor:
        """
        Computes feature vector for the given `node`.

        This vector will be serialized.
        """

class DirectFeature(Feature):
    """
    These features are appended to `torch_geometric.data.Data.x` without further
    processing.
    """

    @property
    @abstractmethod
    def labels(self) -> list[str]:
        """Column names of the resulting feature vector."""

    @property
    def summary(self):
        return { 'labels': self.labels }

class IndirectFeature(Feature):
    """
    These features are appended to `torch_geometric.data.Data` by their `label`.
    """

    @property
    @abstractmethod
    def label(self) -> str:
        """Attribute name on `Data` for the resulting feature vector."""

    @property
    def summary(self):
        return { 'label': self.label }

class Depth(DirectFeature):
    """Relative depth of node in DOM tree."""

    @property
    def labels(self):
        return ['relative_depth']

    @staticmethod
    def _get_max_depth(context: PageContext):
        if context.max_depth is None:
            context.max_depth = max(map(lambda n: n.depth, context.nodes))
        return context.max_depth

    def compute(self, node: 'awe_graph.HtmlNode', context: PageContext):
        return torch.FloatTensor([node.depth / self._get_max_depth(context)])

class IsLeaf(DirectFeature):
    """Whether node is leaf (text) node."""

    @property
    def labels(self):
        return ['is_leaf']

    def compute(self, node: 'awe_graph.HtmlNode', _):
        return torch.FloatTensor([node.is_text])

class CharCategories(DirectFeature):
    """Counts of different character categories."""

    @property
    def labels(self):
        return ['dollars', 'letters', 'digits']

    def compute(self, node: 'awe_graph.HtmlNode', _):
        def count_pattern(pattern: str):
            return len(re.findall(pattern, node.text)) if node.is_text else 0

        return torch.FloatTensor([
            count_pattern(r'[$]'),
            count_pattern(r'[a-zA-Z]'),
            count_pattern(r'\d')
        ])

class FontSize(DirectFeature):
    """Font size (in pixels)."""

    @property
    def labels(self):
        return ['font_size']

    def compute(self, node: 'awe_graph.HtmlNode', _):
        return torch.FloatTensor([node.visual_node.font_size or 0])

def get_default_tokenizer():
    return text_utils.get_tokenizer('basic_english')

class CharIdentifiers(IndirectFeature):
    """Identifiers of characters. Used for randomly-initialized embeddings."""

    def __init__(self):
        self.tokenizer = get_default_tokenizer()

    @property
    def label(self):
        return 'char_identifiers'

    def prepare(self, node: 'awe_graph.HtmlNode', context: RootContext):
        # Find all distinct characters.
        if node.is_text:
            for token in self.tokenizer(node.text):
                context.chars.update(char for char in token)

    def initialize(self, context: LiveContext):
        # Map all founds characters to numbers.
        context.char_dict = { c: i for i, c in enumerate(context.root.chars) }

    def compute(self, node: 'awe_graph.HtmlNode', context: PageContext):
        # Get character indices in each word.
        if node.is_text:
            return [
                torch.IntTensor([
                    context.live.char_dict[char] for char in token
                ])
                for token in self.tokenizer(node.text)
            ]
        return []

class WordIdentifiers(IndirectFeature):
    """Identifiers of word tokens. Used for pre-trained GloVe embeddings."""

    def __init__(self):
        self.tokenizer = get_default_tokenizer()

    @property
    def label(self):
        return 'word_identifiers'

    def prepare(self, node: 'awe_graph.HtmlNode', context: RootContext):
        # Find all distinct tokens.
        if node.is_text:
            context.tokens.update(self.tokenizer(node.text))

    def initialize(self, context: LiveContext):
        # Map all founds tokens to numbers.
        context.token_dict = { t: i for i, t in enumerate(context.root.tokens) }

    def compute(self, node: 'awe_graph.HtmlNode', context: PageContext):
        # Get word token indices.
        if node.is_text:
            return torch.IntTensor([
                context.live.token_dict[token]
                for token in self.tokenizer(node.text)
            ])
        return torch.IntTensor([])
