"""The Apify dataset implementation."""

import dataclasses
import glob
import json
import os
import warnings
from typing import Optional

import ijson
import pandas as pd
import slugify
from tqdm.auto import tqdm

import awe.data.constants
import awe.data.html_utils
import awe.data.parsing
import awe.data.set.db
import awe.data.set.labels
import awe.data.set.pages

DIR = f'{awe.data.constants.DATA_DIR}/apify'
SELECTOR_PREFIX = 'selector_'
VISUALS_SUFFIX = '-exact'
EXACT_EXTENSION = f'{VISUALS_SUFFIX}.html'
PAGES_SUBDIR = 'pages'
LONG_SLUG_WEBSITES = { 'alzaEn', 'bestbuyEn', 'conradEn', 'ikeaEn', 'notinoEn', 'tescoEn' }

@dataclasses.dataclass
class Dataset(awe.data.set.pages.Dataset):
    """The Apify dataset."""

    verticals: list['Vertical'] = dataclasses.field(repr=False)

    def __init__(self,
        only_websites: Optional[list[str]] = None,
        exclude_websites: list[str] = (),
        convert: bool = True,
        convert_slim: bool = False,
        skip_without_visuals: bool = False,
        only_label_keys: Optional[list[str]] = None
    ):
        """
        - `only_websites`: names of websites to load,
        - `exclude_websites`: names of websites to skip,
        - `convert`: save the dataset as SQLite or load existing database,
        - `convert_slim`: save full dataset JSON into a slim version without
          HTML,
        - `skip_without_visuals`: skip pages that have not visuals JSON,
        - `only_label_keys`: see `filter_label_keys`.
        """

        super().__init__(
            name='apify',
            dir_path=DIR,
        )
        self.only_websites = only_websites
        self.exclude_websites = exclude_websites
        self.convert = convert
        self.convert_slim = convert_slim
        self.skip_without_visuals = skip_without_visuals
        self.only_label_keys = only_label_keys
        self.verticals = [
            Vertical(dataset=self, name='product', prev_page_count=0)
        ]

    def filter_label_keys(self, df: pd.DataFrame):
        """
        Transforms the loaded dataset to contain only the specified label keys.
        """

        if (label_keys := self.only_label_keys) is not None:
            all_label_keys = {
                col[len(SELECTOR_PREFIX):]
                for col in df.columns
                if col.startswith(SELECTOR_PREFIX)
            }
            for excluded_label_key in all_label_keys.difference(label_keys):
                del df[excluded_label_key]
                del df[f'{SELECTOR_PREFIX}{excluded_label_key}']

@dataclasses.dataclass
class Vertical(awe.data.set.pages.Vertical):
    dataset: Dataset
    websites: list['Website'] = dataclasses.field(repr=False, default_factory=list)

    def __post_init__(self):
        self.websites = list(self._iterate_websites())

    @property
    def dir_path(self):
        return self.dataset.dir_path

    def get_website_dirs(self):
        """Website names obtained from directories stored on disk."""

        return [
            subdir for subdir in sorted(os.listdir(self.dir_path))
            # Ignore some directories.
            if (subdir not in self.dataset.exclude_websites and
                os.path.isdir(os.path.join(self.dir_path, subdir)) and
                not subdir.startswith('.') and subdir != 'Datasets')
        ]

    def _iterate_websites(self):
        """Construct `Website` instances."""

        if not os.path.exists(self.dir_path):
            warnings.warn(
                f'Dataset directory does not exist ({self.dir_path}).')
            return

        # Get list of websites to load.
        if self.dataset.only_websites is not None:
            website_dirs = self.dataset.only_websites
        else:
            website_dirs = self.get_website_dirs()

        page_count = 0
        for subdir in (p := tqdm(website_dirs, desc='websites')):
            p.set_description(subdir)
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

    db: Optional[awe.data.set.db.Database] = dataclasses.field(repr=False, default=None)
    """SQLite database holding all data for this website."""

    df: Optional[pd.DataFrame] = dataclasses.field(repr=False, default=None)
    """
    The original JSON dataset holding all data except visuals for this website.
    """

    exact_html: bool = dataclasses.field(repr=False, default=False)
    """Whether exact `.html` files have been extracted for this website."""

    def __init__(self, vertical: Vertical, dir_name: str, prev_page_count: int):
        super().__init__(
            vertical=vertical,
            name=dir_name,
            prev_page_count=prev_page_count,
        )

        # Detect whether any extracted exact HTML file exists.
        self.exact_html = any(
            glob.iglob(f'{self.dir_path}/{PAGES_SUBDIR}/*{EXACT_EXTENSION}')
        )

        # Convert if desired.
        if self.vertical.dataset.convert:
            self.create_slim_dataset()
            db = awe.data.set.db.Database(self.dataset_db_path)
            if not db.fresh:
                self.page_count = len(db)
                self.db = db
                self.init_pages()
            else:
                # Load slim dataset if exact HTMLs exist.
                self.load_json_df(slim=self.exact_html)
                self.init_pages()
                if self.vertical.dataset.skip_without_visuals:
                    self.remove_pages_without_visuals()
                    if self.exact_html:
                        self.save_json_df(self.df, self.slim_dataset_json_path)
                self.convert_to_db(db)
            self.load_json_df(slim=True)
        else:
            if self.vertical.dataset.convert_slim:
                self.create_slim_dataset()
                self.load_json_df(slim=True)
            else:
                self.load_json_df(slim=False)
            self.init_pages()
            if self.vertical.dataset.skip_without_visuals:
                self.remove_pages_without_visuals()

    @property
    def dir_path(self):
        return f'{self.vertical.dir_path}/{self.name}'

    @property
    def slim_dataset_json_path(self):
        """
        Path to JSON dataset extracted from the original excluding HTML content.
        """

        return f'{self.dir_path}/slim_dataset.json'

    @property
    def dataset_json_path(self):
        """Path to the original JSON dataset."""

        return f'{self.dir_path}/augmented_dataset.json'

    @property
    def dataset_db_path(self):
        """Path to converted SQLite database."""

        return f'{self.dir_path}/dataset.db'

    @property
    def variable_nodes_file_path(self):
        return f'{self.dir_path}/variable_nodes.txt'

    @property
    def short_slug(self):
        """Whether this website uses short slugs for HTML file names."""

        return self.name not in LONG_SLUG_WEBSITES

    @staticmethod
    def read_json_df(file_path: str):
        """Loads JSON dataset."""

        if not os.path.exists(file_path):
            raise RuntimeError(
                f'JSON not found ({file_path!r}).')
        return pd.read_json(file_path)

    def load_json_df(self, *, slim: bool):
        """Loads JSON (optionally `slim`) dataset."""

        if slim:
            self.df = self.read_json_df(self.slim_dataset_json_path)
        else:
            self.df = self.read_json_df(self.dataset_json_path)
        self.vertical.dataset.filter_label_keys(self.df)
        self.page_count = len(self.df)
        if not slim:
            print(f'Loaded {self.dataset_json_path!r}.')

    def init_pages(self):
        """Construct `Page` instances."""

        self.pages = [
            Page(website=self, index=idx)
            for idx in range(self.page_count)
        ]

    def remove_pages_without_visuals(self):
        """Removes `Page` instances that do not have visuals JSON."""

        remove_indices = [
            p.index for p in self.pages
            if not p.visuals_exist()
        ]
        if len(remove_indices) == 0:
            return
        self.df.drop(index=remove_indices, inplace=True)
        print('Removed pages without visuals ' +
            f'({self.page_count} -> {len(self.df)}).')
        self.page_count = len(self.df)
        self.pages = self.pages[:self.page_count]
        # Re-index.
        self.df.index = range(self.page_count)

    @staticmethod
    def save_json_df(df: pd.DataFrame, file_path: str):
        """Saves JSON dataset to `file_path`."""

        df.to_json(file_path, orient='records')

    def convert_to_db(self, db: awe.data.set.db.Database):
        """Converts JSON dataset to SQLite database."""

        # Add rows to database.
        for page in tqdm(self.pages, desc=self.dataset_db_path):
            page: Page
            db.add(page.index,
                url=page.url,
                html_text=page.get_html_text(),
                visuals=page.create_visuals().get_json_str(),
            )
            if page.index % 100 == 1:
                db.save()
        db.save()

    def create_slim_dataset(self):
        """
        Converts original JSON dataset to a slim version without HTML texts.
        """

        input_path = self.dataset_json_path
        output_path = self.slim_dataset_json_path

        # Skip if the output exists.
        if os.path.exists(output_path):
            return False

        with open(input_path, mode='rb') as input_file:
            with open(output_path, mode='w', encoding='utf-8') as output_file:
                output_file.write('[\n')
                rows = ijson.items(input_file, 'item')
                after_first = False
                for input_row in tqdm(rows, desc=output_path, total=2000):
                    # Keep only `url`, `selector_<key>` and the corresponding
                    # `<key>` columns.
                    label_keys = {
                        k[len(SELECTOR_PREFIX):]
                        for k in input_row.keys()
                        if k.startswith(SELECTOR_PREFIX)
                    }
                    output_row = {
                        k: v
                        for k, v in input_row.items()
                        if (k == 'url' or
                            k.startswith(SELECTOR_PREFIX) or
                            k in label_keys)
                    }
                    if after_first:
                        output_file.write(',\n')
                    else:
                        after_first = True
                    json.dump(output_row, output_file)
                output_file.write(']\n')
        return True

@dataclasses.dataclass(eq=False)
class Page(awe.data.set.pages.Page):
    website: Website
    index: int = None

    @property
    def db(self):
        """SQLite database where this page is stored."""

        return self.website.db

    @property
    def df(self):
        """JSON dataset where this page is stored."""

        return self.website.df

    @property
    def row(self):
        """Row of JSON dataset corresponding to this page."""

        return self.df.iloc[self.index]

    @property
    def metadata(self):
        """Dictionary of metadata for this page such as CSS selectors."""

        return self.row

    @property
    def url_slug(self):
        """URL transformed to construct HTML file name."""

        if self.website.short_slug:
            # HACK: The slug is limited to 100 characters but setting
            # `max_length=100` would sometimes omit the trailing dash.
            return slugify.slugify(self.url, max_length=101)[:100]
        return slugify.slugify(self.url)

    @property
    def file_name_no_extension(self):
        return f'localized_html_{self.url_slug}'

    @property
    def html_file_name(self):
        if self.website.exact_html:
            return f'{self.file_name_no_extension}{EXACT_EXTENSION}'
        return super().html_file_name

    @property
    def visuals_suffix(self):
        return VISUALS_SUFFIX

    @property
    def dir_path(self):
        return f'{self.website.dir_path}/{PAGES_SUBDIR}'

    @property
    def url(self) -> str:
        if self.df is not None:
            return self.row.url
        return self.db.get_url(self.index)

    def get_html_text(self):
        if self.db is not None:
            return self.db.get_html_text(self.index)
        if self.website.exact_html:
            # Exact HTML is not available in the DataFrame, only in file.
            return super().get_html_text()
        return self.row.localizedHtml

    def get_labels(self):
        return PageLabels(self)

    def visuals_exist(self):
        return self.create_visuals().exists()

    def load_visuals(self):
        visuals = self.create_visuals()
        if self.db is not None:
            visuals.load_json_str(self.db.get_visuals(self.index))
        else:
            visuals.load_json()
        return visuals

class PageLabels(awe.data.set.labels.PageLabels):
    """Labels of a `Page` from the Apify dataset."""

    page: Page

    @property
    def label_keys(self):
        keys: list[str] = self.page.metadata.keys()
        return [
            k[len(SELECTOR_PREFIX):]
            for k in keys
            if k.startswith(SELECTOR_PREFIX)
        ]

    def get_selector(self, label_key: str):
        return self.page.metadata[f'{SELECTOR_PREFIX}{label_key}']

    def has_label(self, label_key: str):
        return self.get_selector(label_key) != ''

    def get_label_values(self, label_key: str):
        label_value = self.page.metadata[label_key]

        # Check that when CSS selector is empty string, gold value is empty.
        if not self.has_label(label_key):
            assert label_value == '', \
                f'Unexpected non-empty {label_value=} for {label_key=} ' + \
                f'({self.page.html_path!r}).'
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
            warnings.warn(f'Patched {label_key} selector {selector!r} to ' +
                f'{new_selector!r} in {self.page.website.name!r}.')
            selector = new_selector

        try:
            nodes = self.page.dom.tree.css(selector)
        except awe.data.parsing.Error as e:
            raise RuntimeError(
                f'Invalid selector {selector!r} for {label_key=} ' +
                f'({self.page.html_path!r}).') from e

        # Discard empty labeled nodes (if there are more than 1 labeled).
        if len(nodes) > 1:
            orig_length = len(nodes)
            nodes = [
                n for n in nodes
                if not awe.data.html_utils.is_empty(n)
            ]
            if len(nodes) == 0:
                raise RuntimeError(
                    f'Only empty nodes match {selector=} for {label_key=} ' +
                    f'({self.page.html_path!r}).')
            if orig_length != len(nodes):
                warnings.warn(
                    f'Removed empty nodes labeled {label_key} ({selector=}) ' +
                    f'in {self.page.website.name!r}.')

        return nodes
