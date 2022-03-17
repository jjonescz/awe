import abc
import importlib
import itertools
import os
import sys
from dataclasses import field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, TypeVar

import joblib
import numpy as np
import torch
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from awe import awe_graph


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

R = TypeVar('R')
def parallelize(
    num: Optional[int],
    selector: Callable[[T], R],
    items: Iterable[T],
    desc: str,
    leave: bool = True
) -> list[R]:
    items_with_progress = tqdm(items, desc=desc, leave=leave)
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

def train_val_split(data: list[T], val_split: float):
    split = int(np.floor(val_split * len(data)))
    copy = list(data)
    np.random.seed(42)
    np.random.shuffle(copy)
    return copy[split:], copy[:split]

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

def _summarize_pages(pages: Iterable['awe_graph.HtmlPage']):
    for key, group in itertools.groupby(pages, lambda p: p.group_key):
        ranges = to_ranges(map(lambda p: p.group_index, group))
        yield key, ranges

def summarize_pages(pages: Iterable['awe_graph.HtmlPage']):
    return dict(_summarize_pages(pages))

def sequence_lengths(x: torch.IntTensor):
    """
    Lengths of rows padded with zero.

    Takes `x` shaped `[num_rows, num_cols]`, returns `[num_rows]` with lengths.
    """

    return x.cumsum(dim=1).argmax(dim=1)

def _test_sequence_lengths():
    """Unit test of `sequence_lengths`."""
    x = torch.LongTensor([[1, 0, 3], [1, 2, 0], [1, 0, 0]])
    y = sequence_lengths(x)
    assert torch.equal(y, torch.LongTensor([2, 1, 0]))

_test_sequence_lengths()

def copy_to(source: torch.Tensor, target: torch.Tensor):
    """
    Copies cells from `source` to `target` ignoring indices they don't share.
    """

    shape = np.minimum(source.shape, target.shape)
    indices = tuple(slice(None, i) for i in shape)
    target[indices] = source[indices]

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

def find_all(string: str, substring: str, overlap: bool = False):
    """Finds all `substring` occurrences (start indices) in `string`."""
    return list(_iterate_all(
        string=string,
        substring=substring,
        overlap=overlap
    ))

def _iterate_all(string: str, substring: str, overlap: bool):
    start = 0
    while True:
        start = string.find(substring, start)
        if start < 0:
            break
        yield start
        start += 1 if overlap else len(substring)

def at_index(collection: Iterable[T], idx: int):
    return next(itertools.islice(collection, idx, None))

T1 = TypeVar('T1')
T2 = TypeVar('T2')
def unzip(collection: Iterable[tuple[T1, T2]]):
    return tuple(zip(*collection))

def change_dir_to_project_root():
    while not os.path.exists('awe'):
        print(os.getcwd())
        os.chdir('..')
    print(os.getcwd())

def test_tqdm():
    _ = list(tqdm(range(1)))

def init_notebook():
    change_dir_to_project_root()
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
