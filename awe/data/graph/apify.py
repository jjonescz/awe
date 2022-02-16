import dataclasses
import os
import warnings

import pandas as pd
import slugify

import awe.data.constants
import awe.data.graph.labels
import awe.data.graph.pages
import awe.data.parsing

DIR = f'{awe.data.constants.DATA_DIR}/apify'
SELECTOR_PREFIX = 'selector_'

@dataclasses.dataclass
class Dataset(awe.data.graph.pages.Dataset):
    verticals: list['Vertical'] = dataclasses.field(repr=False)

    def __init__(self):
        super().__init__(
            name='apify',
            dir_path=DIR,
        )
        self.verticals = [
            Vertical(dataset=self, name='products')
        ]

@dataclasses.dataclass
class Vertical(awe.data.graph.pages.Vertical):
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

        for subdir in sorted(os.listdir(self.dir_path)):
            yield Website(self, subdir)

@dataclasses.dataclass
class Website(awe.data.graph.pages.Website):
    vertical: Vertical
    df: pd.DataFrame = dataclasses.field(repr=False, default=None)

    def __init__(self, vertical: Vertical, dir_name: str):
        super().__init__(
            vertical=vertical,
            name=dir_name,
        )

        # Convert dataset.
        if not os.path.exists(self.dataset_pickle_path):
            print('Saving dataset in efficient binary format ' + \
                f'({self.dataset_pickle_path}).')
            json_df = pd.read_json(self.dataset_json_path)
            json_df.to_pickle(self.dataset_pickle_path)

        # Load dataset.
        self.df = pd.read_pickle(self.dataset_pickle_path)
        self.pages = [
            Page(website=self, index=idx)
            for idx in range(len(self.df))
        ]

    @property
    def dir_path(self):
        return f'{self.vertical.dir_path}/{self.name}'

    @property
    def dataset_json_path(self):
        return f'{self.dir_path}/dataset.json'

    @property
    def dataset_pickle_path(self):
        return f'{self.dir_path}/dataset.pkl'

@dataclasses.dataclass
class Page(awe.data.graph.pages.Page):
    website: Website
    index: int

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

class PageLabels(awe.data.graph.labels.PageLabels):
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
        if not self.has_label(label_key):
            assert label_value == '', \
                f'Unexpected non-empty {label_value=} for {label_key=}.'
            return []
        return [label_value]

    def get_labeled_nodes(self, label_key: str):
        if not self.has_label(label_key):
            return []
        selector = self.get_selector(label_key)
        try:
            return self.dom.tree.css(selector)
        except awe.data.parsing.Error as e:
            page_html_path = self.page.html_path
            raise RuntimeError(
                f'Invalid selector {repr(selector)} for {label_key=} ' + \
                f'({page_html_path}).') from e
