"""The SWDE dataset implementation."""

import abc
import dataclasses
import glob
import os
import re
import warnings
from typing import Optional

from tqdm.auto import tqdm

import awe.data.constants
import awe.data.set.db
import awe.data.set.pages
import awe.data.set.swde_labels
import awe.utils

DIR = f'{awe.data.constants.DATA_DIR}/swde'
DATA_DIR = f'{DIR}/data'
VERTICAL_NAMES = [
    'auto',
    'book',
    'camera',
    'job',
    # HACK: Skip movie vertical as there are multiple bugs in the dataset:
    # - MPAA rating is only first character (e.g., "P" in the groundtruth but
    #   "PG13" in the HTML),
    # - director is not complete (e.g., "Roy Hill" in the groundtruth but
    #   "George Roy Hill" in the HTML).
    #'movie',
    'nbaplayer',
    'restaurant',
    'university'
]
WEBSITE_REGEX = r'^(\w+)-(\w+)\((\d+)\)$'
PAGE_REGEX = r'^(\d{4})(-.*)?\.htm$'
BASE_TAG_REGEX = r'^<base href="([^\n]*)"/>\w*\n$'

@dataclasses.dataclass
class Dataset(awe.data.set.pages.Dataset):
    verticals: list['Vertical'] = dataclasses.field(repr=False)

    suffix: Optional[str] = None
    """
    Suffix corresponding to outputs of the visual extractor (e.g., `-exact`).
    """

    def __init__(self,
        suffix: Optional[str] = None,
        only_verticals: Optional[list[str]] = None,
        only_websites: Optional[list[str]] = None,
        convert: bool = True
    ):
        """
        - `only_verticals`: names of verticals to load,
        - `only_websites`: names of websites to load,
        - `convert`: save the dataset as SQLite or load existing database.
        """

        super().__init__(
            name=f'swde{suffix or ""}',
            dir_path=DATA_DIR,
        )
        self.suffix = suffix
        self.only_verticals = only_verticals
        self.only_websites = only_websites
        self.convert = convert
        self.verticals = list(self._iterate_verticals())

    def _iterate_verticals(self):
        """Constructs `Vertical` instances."""

        # Get list of verticals to load.
        if self.only_verticals:
            vertical_names = self.only_verticals
        else:
            vertical_names = VERTICAL_NAMES

        page_count = 0
        for name in tqdm(vertical_names, desc='verticals'):
            vertical = Vertical(
                dataset=self,
                name=name,
                prev_page_count=page_count,
            )
            yield vertical
            page_count += vertical.page_count

    def find_page(self, vertical: str, website: str, page: int):
        """
        Finds page given its `vertical` name, `website` name and `page` index.
        """

        v = next(v for v in self.verticals if v.name == vertical)
        w = next(w for w in v.websites if w.name == website)
        return w.pages[page]

@dataclasses.dataclass
class Vertical(awe.data.set.pages.Vertical):
    dataset: Dataset
    websites: list['Website'] = dataclasses.field(repr=False, default_factory=list)

    db: awe.data.set.db.Database = dataclasses.field(repr=False, default=None)
    """SQLite database holding all data for this website."""

    def __post_init__(self):
        if not os.path.exists(self.dir_path):
            warnings.warn(
                f'Vertical directory does not exist ({self.dir_path!r}).')
            self.page_count = 0
            return

        # Fill database.
        self.db = awe.data.set.db.Database(self.db_path)
        if self.dataset.convert and self.db.fresh:
            # Convert vertical to an SQLite database. It is faster than reading
            # from files (especially in the cloud where the files are
            # usually in a mounted file system which can be really slow).
            pages = [
                page
                for website in self._iterate_websites(FileWebsite)
                for page in website.pages
            ]
            for idx, page in enumerate(tqdm(pages, desc=self.db_path)):
                page: FilePage
                self.db.add(page.index_in_vertical, **page.to_row())
                if idx % 100 == 1:
                    self.db.save()
            self.db.save()

        if not self.dataset.convert:
            self.websites = list(self._iterate_websites(FileWebsite))
            self.page_count = sum(w.page_count for w in self.websites)
        else:
            self.websites = list(self._iterate_websites(DbWebsite))
            self.page_count = len(self.db)

    @property
    def dir_path(self):
        return f'{self.dataset.dir_path}/{self.name}'

    @property
    def db_path(self):
        return f'{self.dir_path}{self.dataset.suffix or ""}.db'

    @property
    def groundtruth_dir(self):
        """Directory containing `GroundtruthFile`s."""

        return f'{self.dataset.dir_path}/groundtruth/{self.name}'

    @property
    def groundtruth_path_prefix(self):
        """Path to `GroundtruthFile`s of this vertical."""

        return f'{self.groundtruth_dir}/{self.name}'

    def _iterate_websites(self, cls: type['Website']):
        """Constructs `Website` instances of the specified type (`cls`)."""

        # Get list of websites to load.
        website_dirs = sorted(os.listdir(self.dir_path))
        if self.dataset.only_websites is not None:
            website_dirs = [
                d for d in website_dirs
                if any(
                    d.startswith(f'{self.name}-{w}(')
                    for w in self.dataset.only_websites
                )
            ]

        page_count = 0
        for subdir in website_dirs:
            website = cls(
                vertical=self,
                dir_name=subdir,
                prev_page_count=page_count
            )
            assert website.dir_name == subdir

            # HACK: Skip website careerbuilder.com whose groundtruth values are
            # in HTML comments (that's a bug in the dataset).
            if self.name == 'job' and website.name == 'careerbuilder':
                continue

            yield website
            page_count += website.page_count

@dataclasses.dataclass
class Website(awe.data.set.pages.Website):
    vertical: Vertical
    pages: list['Page'] = dataclasses.field(repr=False)

    groundtruth_files: dict[str, awe.data.set.swde_labels.GroundtruthFile] = \
        dataclasses.field(repr=False, default=None)
    """`GroundtruthFile`s for attribute keys of this website."""

    def __init__(self, vertical: Vertical, dir_name: str, prev_page_count: int):
        match = re.search(WEBSITE_REGEX, dir_name)
        assert vertical.name == match.group(1)
        super().__init__(
            vertical=vertical,
            name=match.group(2),
            prev_page_count=prev_page_count,
            page_count=int(match.group(3))
        )
        self.groundtruth_files = {
            g.label_key: g for g in self._iterate_groundtruth()
        }
        self.pages = list(self._iterate_pages())

        if len(self.pages) != self.page_count:
            existing = {p.index for p in self.pages}
            missing = [
                idx for idx in range(self.page_count)
                if idx not in existing
            ]
            warnings.warn('Some pages were not created for site ' +
                f'{self.dir_name} ({awe.utils.to_ranges(missing)}).')

    @property
    def dir_name(self):
        return f'{self.vertical.name}-{self.name}({self.page_count})'

    @property
    def dir_path(self):
        return f'{self.vertical.dir_path}/{self.dir_name}'

    @property
    def variable_nodes_file_path(self):
        return f'{self.dir_path}/variable_nodes.txt'

    @property
    def groundtruth_path_prefix(self):
        """Path to `GroundtruthFile`s of this website."""

        return f'{self.vertical.groundtruth_path_prefix}-{self.name}'

    @abc.abstractmethod
    def _iterate_pages(self):
        """Constructs `Page` instances."""

    def _iterate_groundtruth(self):
        """Constructs `GroundtruthFile` instances."""

        for file in glob.glob(f'{self.groundtruth_path_prefix}-*.txt'):
            file_name = os.path.basename(file)
            groundtruth = awe.data.set.swde_labels.GroundtruthFile(
                website=self,
                file_name=file_name,
            )
            assert os.path.samefile(groundtruth.file_path, file)
            yield groundtruth

    def get_page_at(self, idx: int):
        """Finds page at the specified `idx`."""

        # Hot path: try indexing directly.
        if idx < len(self.pages) and (page := self.pages[idx]).index == idx:
            return page

        # Slow path: find the page manually.
        for page in self.pages:
            if page.index == idx:
                return page
        return None

class FileWebsite(Website):
    """SWDE website with pages stored in the original files."""

    def _iterate_pages(self):
        for file in sorted(os.listdir(f'{self.dir_path}')):
            page = FilePage.try_create(website=self, file_name=file)
            if page is not None:
                assert page.html_file_name == file
                yield page

class DbWebsite(Website):
    """SWDE website with pages stored in the SQLite database."""

    def _iterate_pages(self):
        for idx in range(self.page_count):
            yield DbPage(website=self, index=idx)

@dataclasses.dataclass(eq=False)
class Page(awe.data.set.pages.Page):
    website: Website

    _url: Optional[str] = dataclasses.field(repr=False, init=False, default=None)
    """Original URL of the page."""

    groundtruth_entries: dict[str, awe.data.set.swde_labels.GroundtruthEntry] = \
        dataclasses.field(repr=False, default=None)
    """`GroundtruthEntry`s for attribute keys of this page."""

    def __post_init__(self):
        self.groundtruth_entries = {
            f.label_key: f.get_entry_for(self)
            for f in self.website.groundtruth_files.values()
        }

    @property
    def suffix(self):
        return self.website.vertical.dataset.suffix

    @property
    def original_file_name_no_extension(self):
        return f'{self.index:04}'

    @property
    def file_name_no_extension(self):
        return f'{self.original_file_name_no_extension}{self.suffix or ""}'

    @property
    def dir_path(self):
        return self.website.dir_path

    def get_labels(self):
        return awe.data.set.swde_labels.PageLabels(self)

    def get_visuals_json_text(self):
        return self.create_visuals().get_json_str()

    def to_row(self):
        """
        Constructs dictionary with page metadata (to be stored in the SQLite
        database).
        """

        return {
            'url': self.url,
            'html_text': self.get_html_text(),
            'visuals': self.get_visuals_json_text()
        }

    def reload(self):
        """Reloads this page in the database from the original file."""

        file_page = FilePage.try_create(self.website, self.html_file_name)
        self.website.pages[self.index] = file_page
        self.website.vertical.db.replace(self.index_in_vertical, **file_page.to_row())
        return file_page

class FilePage(Page):
    """Page inside `FileWebsite`."""

    @staticmethod
    def try_create(website: Website, file_name: str):
        """Creates instance if `file_name` matches the correct pattern."""

        match = re.search(PAGE_REGEX, file_name)
        if match is None:
            return None
        page = FilePage(website=website, index=int(match.group(1)))
        suffix = match.group(2)
        if suffix != page.suffix:
            return None

        return page

    @property
    def url(self):
        if self._url is None:
            # Note that there is a `<base />` tag appended before each HTML
            # document in SWDE with the actual crawled URL.
            with open(self.html_path, 'r', encoding='utf-8-sig') as f:
                link = f.readline()
            match = re.search(BASE_TAG_REGEX, link)
            self._url = match.group(1)
        return self._url

    def get_html_text(self):
        with open(self.html_path, 'r', encoding='utf-8-sig') as f:
            # Discard first line with URL.
            _ = f.readline()
            return f.read()

    def load_visuals(self):
        visuals = self.create_visuals()
        visuals.load_json()
        return visuals

class DbPage(Page):
    """Page inside `DbWebsite`."""

    @property
    def db(self):
        """SQLite database where this page is stored."""

        return self.website.vertical.db

    @property
    def url(self):
        return self.db.get_url(self.index_in_vertical)

    def get_html_text(self):
        return self.db.get_html_text(self.index_in_vertical)

    def get_visuals_json_text(self):
        return self.db.get_visuals(self.index_in_vertical)

    def load_visuals(self):
        visuals = self.create_visuals()
        visuals.load_json_str(self.get_visuals_json_text())
        return visuals
