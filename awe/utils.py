import os
from dataclasses import field
from typing import Any, Callable, Iterable, Optional, TypeVar

import joblib
from tqdm.auto import tqdm


def add_field(**kwargs):
    """Additional field (not contributing to identity)."""
    return field(hash=False, compare=False, **kwargs)

def ignore_field(**kwargs):
    return add_field(init=False, repr=False, **kwargs)

def cache_field(**kwargs):
    return ignore_field(default=None, **kwargs)

def lazy_field(**kwargs):
    """Field initialized lazily (later during instance lifetime)."""
    return field(init=False, default=None, **kwargs)

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

def parallelize(
    num: Optional[int],
    selector: Callable[[T], Any],
    items: Iterable[T],
    desc: str
):
    items_with_progress =  tqdm(items, desc=desc)
    if num is None:
        return list(map(selector, items_with_progress))
    return list(joblib.Parallel(n_jobs=num)(
        map(joblib.delayed(selector), items_with_progress)))

def save_or_check_file(path: str, content: str):
    """
    Saves `contents` to `file` if it doesn't exist yet. Otherwise, checks that
    the existing file has the same content.
    """
    if os.path.exists(path):
        # If the file already exists, check that it has the same contents.
        with open(path, mode='r', encoding='utf-8') as file:
            if file.read() != content:
                raise RuntimeError(
                    f'File content at {path} is different from expected: ' + \
                    f'{content}')
    else:
        with open(path, mode='w', encoding='utf-8') as file:
            file.write(content)
    return path
