"""Shared context used when computing visual features."""

import collections
import dataclasses
from typing import Callable

@dataclasses.dataclass
class CategoryInfo:
    """Represents one category of a categorical feature."""

    unique_id: int
    """
    Unique ID of the category.

    Starts from 1, because 0 is reserved for the unknown category.
    """

    count: int
    """How many times was this category used."""

    def merge_with(self, other: 'CategoryInfo'):
        self.count += other.count

def categorical_dict() -> dict[str, CategoryInfo]:
    """Creates `defaultdict` of `CategoryInfo`s."""

    d = collections.defaultdict()
    # Note that ID `0` is reserved for "unseen" category.
    d.default_factory = lambda: CategoryInfo(len(d) + 1, 0)
    return d

def update_values(
    name: str,
    values: list[float],
    d: dict[str, list[float]],
    f: Callable[[float, float], float]
):
    """Updates `min_values` or `max_values` of `Extraction`."""

    d_values = d.setdefault(name, values)
    d[name] = [f(a, b) for a, b in zip(d_values, values)]

class Extraction:
    """Shared context for visual feature extraction."""

    categorical: collections.defaultdict[str,
        collections.defaultdict[str, CategoryInfo]]
    """
    Visual categorical features (name of feature -> category label -> category
    info).
    """

    min_values: dict[str, list[float]]
    max_values: dict[str, list[float]]

    def __init__(self):
        self.categorical = collections.defaultdict(categorical_dict)
        self.min_values = {}
        self.max_values = {}

    def update_values(self, name: str, values: list[float]):
        """Updates min-max values."""

        update_values(name, values, self.min_values, min)
        update_values(name, values, self.max_values, max)

    def describe(self):
        """Constructs dictionary representing this context."""

        return self.describe_categorical() | self.describe_min_max() | {
            'total_categorical': self.total_categorical_count()
        }

    def describe_categorical(self):
        """Determines number of values for each category of each feature."""

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
        """Counts total number of values of categorical features."""

        return sum(
            i.count
            for c in self.categorical.values()
            for i in c.values()
        )

    def describe_min_max(self):
        """Describes ranges of min-max visual features."""

        return {
            k: [self.min_values.get(k), self.max_values.get(k)]
            for k in self.min_values.keys() | self.max_values.keys()
        }

    def freeze(self):
        """Freezes this object for pickling."""

        for value in self.categorical.values():
            value.default_factory = None
        self.categorical.default_factory = None
