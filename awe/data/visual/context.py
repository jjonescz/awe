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

class Extraction:
    """Shared context for visual feature extraction."""

    categorical: collections.defaultdict[str,
        collections.defaultdict[str, CategoryInfo]]
    """
    Visual categorical features (name of feature -> category label -> category
    info).
    """

    def __init__(self):
        self.categorical = collections.defaultdict(categorical_dict)

    def describe_categorical(self):
        return {
            # Place most used first.
            feature: dict(sorted(
                category.items(),
                key=lambda p: p[1].count,
                reverse=True
            ))
            for feature, category in self.categorical.items()
        }

    def total_categorical_count(self):
        return sum(
            i.count
            for c in self.categorical.values()
            for i in c.values()
        )

    def freeze(self):
        # Needed to pickle this object.
        for value in self.categorical.values():
            value.default_factory = None
        self.categorical.default_factory = None
