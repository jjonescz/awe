import collections
import dataclasses


@dataclasses.dataclass
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

    html_tags: set[str]

    html_tag_ids: dict[str, int]

    max_word_len: int = 0
    """Length of the longest word. Stored by `CharacterIdentifiers`."""

    max_num_words: int = 0
    """
    Number of words in the longest node (up to `cutoff_words`). Stored by
    `CharacterIdentifiers` and `WordIdentifiers`.
    """

    char_dict: dict[str, int]
    """Used by `CharacterEmbedding`."""

    token_dict: dict[str, int]
    """Used by `WordEmbedding`."""

    visual_categorical: collections.defaultdict[str,
        collections.defaultdict[str, CategoryInfo]]
    """
    Visual categorical features (name of feature -> category label -> category
    info). Used by `Visuals`.
    """

    def __init__(self):
        self.pages = set()
        self.chars = set()
        self.html_tags = set()
        self.char_dict = {}
        self.word_dict = {}
        self.visual_categorical = collections.defaultdict(categorical_dict)

    def merge_with(self, other: 'RootContext'):
        self.pages.update(other.pages)
        self.chars.update(other.chars)
        self.max_word_len = max(self.max_word_len, other.max_word_len)
        self.max_num_words = max(self.max_num_words, other.max_num_words)

        # TODO: Merge all attributes.

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
