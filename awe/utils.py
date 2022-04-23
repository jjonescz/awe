"""Common utilities."""

import abc
import importlib
import itertools
import sys
from typing import Any, Callable, Iterable, TypeVar

from tqdm.auto import tqdm

T = TypeVar('T')
def where_max(items: Iterable[T], selector: Callable[[T], Any]):
    max_key = None
    max_item = None
    for item in items:
        key = selector(item)
        if max_key is None:
            max_key = key
            max_item = item
        elif key > max_key:
            max_key = key
            max_item = item
    return max_item

def _iterate_ranges(iterable: Iterable[T]):
    # Inspired by https://stackoverflow.com/a/43091576.
    iterable = sorted(set(iterable))
    for _, group in itertools.groupby(
        enumerate(iterable),
        lambda t: t[1] - t[0]
    ):
        group = list(group)
        yield group[0][1], group[-1][1]

def to_ranges(iterable: Iterable[T]):
    return list(_iterate_ranges(iterable))

def to_camel_case(snake_case: str):
    # Inspired by https://stackoverflow.com/a/19053800.
    parts = snake_case.split('_')
    return parts[0] + ''.join(p.title() for p in parts[1:])

def reload(*modules: list[str], exclude: list[str] = ()):
    for module in modules:
        # Inspired by https://stackoverflow.com/a/51074507.
        for k, v in list(sys.modules.items()):
            if k.startswith(module) and not any(k.startswith(e) for e in exclude):
                importlib.reload(v)

def test_tqdm():
    _ = list(tqdm(range(1)))

def init_notebook():
    test_tqdm()

def full_name(cls: type):
    return f'{cls.__module__}.{cls.__name__}'

def same_types(a: type, b: type):
    return full_name(a) == full_name(b)

def get_attrs(obj, attrs: list[str]):
    return {
        a: getattr(obj, a)
        for a in attrs
    }

def set_attrs(obj, attrs: dict[str]):
    for k, v in attrs.items():
        setattr(obj, k, v)

class PickleSubset(abc.ABC):
    """
    Derive from this mixin to easily support pickling a subset of a class's
    fields.
    """

    @abc.abstractmethod
    def get_pickled_keys(self) -> list[str]:
        """List of keys to (un)pickle."""

    def __getstate__(self):
        return get_attrs(self, self.get_pickled_keys())

    def __setstate__(self, state):
        set_attrs(self, state)
