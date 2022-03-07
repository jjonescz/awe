import dataclasses
import os
import warnings
from typing import Optional

import pandas as pd
import slugify
from tqdm.auto import tqdm

import awe.data.constants
import awe.data.parsing
import awe.data.set.labels
import awe.data.set.pages

DIR = f'{awe.data.constants.DATA_DIR}/apify'
SELECTOR_PREFIX = 'selector_'

@dataclasses.dataclass
class Dataset(awe.data.set.pages.Dataset):
    verticals: list['Vertical'] = dataclasses.field(repr=False)

    def __init__(self, only_websites: Optional[list[str]] = None):
        super().__init__(
            name='apify',
            dir_path=DIR,
        )
        self.only_websites = only_websites
        self.verticals = [
            Vertical(dataset=self, name='products', prev_page_count=0)
        ]

@dataclasses.dataclass
class Vertical(awe.data.set.pages.Vertical):
    dataset: Dataset
    websites: list['Website'] = dataclasses.field(repr=False, default_factory=list)

    def __post_init__(self):
        self.websites = list(self._iterate_websites())

    @property
    def dir_path(self):
        return self.dataset.dir_path

    def _iterate_websites(self):
        if not os.path.exists(self.dir_path):
            warnings.warn(
                f'Dataset directory does not exist ({self.dir_path}).')
            return

        page_count = 0
        for subdir in tqdm(sorted(os.listdir(self.dir_path)), desc='websites'):
            if (self.dataset.only_websites is not None
                and subdir not in self.dataset.only_websites):
                continue

            website = Website(
                vertical=self,
                dir_name=subdir,
                prev_page_count=page_count
            )
            yield website
            page_count += website.page_count

@dataclasses.dataclass
class Website(awe.data.set.pages.Website):
    vertical: Vertical
    df: pd.DataFrame = dataclasses.field(repr=False, default=None)

    def __init__(self, vertical: Vertical, dir_name: str, prev_page_count: int):
        super().__init__(
            vertical=vertical,
            name=dir_name,
            prev_page_count=prev_page_count,
        )

        # Convert dataset.
        if not os.path.exists(self.dataset_pickle_path):
            print('Saving dataset in efficient binary format ' + \
                f'({self.dataset_pickle_path!r}).')

            if not os.path.exists(self.dataset_json_path):
                raise RuntimeError(
                    f'JSON not found ({self.dataset_json_path!r}).')

            json_df = pd.read_json(self.dataset_json_path)
            json_df.to_pickle(self.dataset_pickle_path)

        # Load dataset.
        self.df = pd.read_pickle(self.dataset_pickle_path)
        self.pages = [
            Page(website=self, index=idx)
            for idx in range(len(self.df))
        ]
        self.page_count = len(self.pages)

    @property
    def dir_path(self):
        return f'{self.vertical.dir_path}/{self.name}'

    @property
    def dataset_json_path(self):
        return f'{self.dir_path}/dataset.json'

    @property
    def dataset_pickle_path(self):
        return f'{self.dir_path}/dataset.pkl'

@dataclasses.dataclass(eq=False)
class Page(awe.data.set.pages.Page):
    website: Website
    index: int = None

    @property
    def row(self):
        return self.website.df.iloc[self.index]

    @property
    def url_slug(self):
        return slugify.slugify(self.url)

    @property
    def html_file_name(self):
        return f'original_html_{self.url_slug}.htm'

    @property
    def html_path(self):
        return f'{self.website.dir_path}/pages/{self.html_file_name}'

    @property
    def url(self) -> str:
        return self.row.url

    def get_html_text(self):
        return self.row.html

    def get_labels(self):
        return PageLabels(self)

class PageLabels(awe.data.set.labels.PageLabels):
    page: Page

    @property
    def label_keys(self):
        keys: list[str] = self.page.row.keys()
        return [
            k[len(SELECTOR_PREFIX):]
            for k in keys
            if k.startswith(SELECTOR_PREFIX)
        ]

    def get_selector(self, label_key: str):
        return self.page.row[f'{SELECTOR_PREFIX}{label_key}']

    def has_label(self, label_key: str):
        return self.get_selector(label_key) != ''

    def get_label_values(self, label_key: str):
        label_value = self.page.row[label_key]

        # Check that when CSS selector is empty string, gold value is empty.
        if not self.has_label(label_key):
            assert label_value == '', \
                f'Unexpected non-empty {label_value=} for {label_key=}.'
            return []

        # HACK: Sometimes in the dataset, the node does not exist even though it
        # has a selector specified. Then we don't want to return `['']` (one
        # empty node), but `[]` (no nodes) instead. Prerequisite for this
        # situation is that the value is empty (but it can be string or list).
        if not label_value and len(self.get_labeled_nodes(label_key)) == 0:
            selector = self.get_selector(label_key)
            warnings.warn(f'Ignoring non-existent {selector=} for ' + \
                f'{label_key=} ({self.page.url}).')
            return []

        return [label_value]

    def get_labeled_nodes(self, label_key: str):
        if not self.has_label(label_key):
            return []
        selector = self.get_selector(label_key)

        # HACK: If selector contains `+`, replace it by `~` as there is a bug in
        # Lexbor's implementation of the former (a segfault occurs in
        # `lxb_selectors_sibling` at source/lexbor/selectors/selectors.c:266).
        if '+' in selector:
            new_selector = selector.replace('+', '~')
            warnings.warn(f'Patched selector {selector!r} to {new_selector!r}.')
            selector = new_selector

        try:
            return self.page.dom.tree.css(selector)
        except awe.data.parsing.Error as e:
            raise RuntimeError(
                f'Invalid selector {repr(selector)} for {label_key=} ' + \
                f'({self.page.url}).') from e
