import dataclasses
import glob
import json
import os
import warnings
from typing import Optional

import ijson
import json5
import pandas as pd
import slugify
from tqdm.auto import tqdm

import awe.data.constants
import awe.data.parsing
import awe.data.set.db
import awe.data.set.labels
import awe.data.set.pages

DIR = f'{awe.data.constants.DATA_DIR}/apify'
SELECTOR_PREFIX = 'selector_'
VISUALS_SUFFIX = '-exact'
EXACT_EXTENSION = f'{VISUALS_SUFFIX}.html'
PAGES_SUBDIR = 'pages'

@dataclasses.dataclass
class Dataset(awe.data.set.pages.Dataset):
    verticals: list['Vertical'] = dataclasses.field(repr=False)

    def __init__(self,
        only_websites: Optional[list[str]] = None,
        exclude_websites: list[str] = (),
        convert: bool = True,
        only_label_keys: Optional[list[str]] = None
    ):
        super().__init__(
            name='apify',
            dir_path=DIR,
        )
        self.only_websites = only_websites
        self.exclude_websites = exclude_websites
        self.convert = convert
        self.only_label_keys = only_label_keys
        self.verticals = [
            Vertical(dataset=self, name='products', prev_page_count=0)
        ]

    def filter_label_keys(self, df: pd.DataFrame):
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
        return [
            subdir for subdir in sorted(os.listdir(self.dir_path))
            # Ignore some directories.
            if (subdir not in self.dataset.exclude_websites and
                os.path.isdir(os.path.join(self.dir_path, subdir)) and
                not subdir.startswith('.') and subdir != 'Datasets')
        ]

    def _iterate_websites(self):
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
    df: Optional[pd.DataFrame] = dataclasses.field(repr=False, default=None)

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

        # Convert if desired and no exact HTML exists (otherwise, conversion
        # would be counter-productive).
        if self.vertical.dataset.convert and not self.exact_html:
            self.create_slim_dataset()
            db = awe.data.set.db.Database(self.dataset_db_path)
            if not db.fresh:
                self.page_count = len(db)
                self.db = db
                self.init_pages()
            else:
                self.load_json_df(slim=False)
                self.init_pages()
                self.convert_to_db(db)
            self.load_json_df(slim=True)
        else:
            if self.vertical.dataset.convert and self.exact_html:
                self.create_slim_dataset()
                self.load_json_df(slim=True)
            else:
                self.load_json_df(slim=False)
            self.init_pages()

    @property
    def dir_path(self):
        return f'{self.vertical.dir_path}/{self.name}'

    @property
    def slim_dataset_json_path(self):
        return f'{self.dir_path}/slim_dataset.json'

    @property
    def dataset_json_path(self):
        return f'{self.dir_path}/augmented_dataset.json'

    @property
    def dataset_db_path(self):
        return f'{self.dir_path}/dataset.db'

    @staticmethod
    def read_json_df(file_path: str):
        if not os.path.exists(file_path):
            raise RuntimeError(
                f'JSON not found ({file_path!r}).')
        return pd.read_json(file_path)

    def load_json_df(self, *, slim: bool):
        if slim:
            self.df = self.read_json_df(self.slim_dataset_json_path)
        else:
            self.df = self.read_json_df(self.dataset_json_path)
        self.vertical.dataset.filter_label_keys(self.df)
        self.page_count = len(self.df)
        if not slim:
            print(f'Loaded {self.dataset_json_path!r}.')

    def init_pages(self):
        self.pages = [
            Page(website=self, index=idx)
            for idx in range(self.page_count)
        ]

    @staticmethod
    def save_json_df(df: pd.DataFrame, file_path: str):
        df.to_json(file_path, orient='records')

    def convert_to_db(self, db: awe.data.set.db.Database):
        # Gather DataFrame columns to convert into metadata.
        selector_cols = {
            col for col in self.df.columns
            if col.startswith(SELECTOR_PREFIX)
        }
        metadata_cols = selector_cols | {
            col[len(SELECTOR_PREFIX):]
            for col in selector_cols
        }

        # Add rows to database.
        for page in tqdm(self.pages, desc=self.dataset_db_path):
            page: Page
            metadata_dict = {
                k: v
                for k, v in page.metadata.items()
                if k in metadata_cols
            }
            metadata_json = json5.dumps(metadata_dict)
            db.add(page.index,
                url=page.url,
                html_text=page.get_html_text(),
                visuals=page.get_visuals_json_text(),
                metadata=metadata_json
            )
            if page.index % 100 == 1:
                db.save()
        db.save()

    def create_slim_dataset(self):
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
        return self.website.db

    @property
    def df(self):
        return self.website.df

    @property
    def row(self):
        return self.df.iloc[self.index]

    @property
    def metadata(self):
        if self.df is not None:
            return self.row
        json_text = self.db.get_metadata(self.index)
        return json5.loads(json_text)

    @property
    def url_slug(self):
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

    def get_visuals_json_text(self):
        if self.db is not None:
            return self.db.get_visuals(self.index)
        return self.create_visuals().get_json_str()

    def load_visuals(self):
        visuals = self.create_visuals()
        visuals.load_json_str(self.get_visuals_json_text())
        return visuals

class PageLabels(awe.data.set.labels.PageLabels):
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
                f'{new_selector!r}.')
            selector = new_selector

        try:
            nodes = self.page.dom.tree.css(selector)
        except awe.data.parsing.Error as e:
            raise RuntimeError(
                f'Invalid selector {selector!r} for {label_key=} ' + \
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
                    f'Removed empty nodes labeled {label_key} ({selector=}).')

        return nodes
