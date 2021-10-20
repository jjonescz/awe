from dataclasses import field
from typing import Any, Callable, Iterable, TypeVar


def add_field(**kwargs):
    return field(hash=False, compare=False, **kwargs)

def ignore_field(**kwargs):
    return add_field(init=False, repr=False, **kwargs)

def cache_field(**kwargs):
    return ignore_field(default=None, **kwargs)

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
