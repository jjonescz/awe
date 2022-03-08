import abc
import dataclasses
import glob
import os
import re
import warnings
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

import awe.data.constants
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

    def __init__(self,
        suffix: Optional[str] = None,
        only_verticals: Optional[list[str]] = None,
        state: Optional[dict[str, pd.DataFrame]] = None
    ):
        super().__init__(
            name=f'swde{suffix or ""}',
            dir_path=DATA_DIR,
        )
        self.suffix = suffix
        self.only_verticals = only_verticals
        self.verticals = list(self._iterate_verticals(state=state or {}))

    def _iterate_verticals(self, state: dict[str, pd.DataFrame]):
        page_count = 0
        for name in tqdm(VERTICAL_NAMES, desc='verticals'):
            if (self.only_verticals is not None
                and name not in self.only_verticals):
                continue

            vertical = Vertical(
                dataset=self,
                name=name,
                prev_page_count=page_count,
                df=state.get(name),
            )
            yield vertical
            page_count += vertical.page_count

    def get_state(self):
        return { v.name: v.df for v in self.verticals }

@dataclasses.dataclass
class Vertical(awe.data.set.pages.Vertical):
    dataset: Dataset
    websites: list['Website'] = dataclasses.field(repr=False, default_factory=list)
    df: pd.DataFrame = dataclasses.field(repr=False, default=None)

    def __post_init__(self):
        if not os.path.exists(self.dir_path):
            warnings.warn(
                f'Vertical directory does not exist ({self.dir_path!r}).')
            self.page_count = 0
            return

        if self.df is not None:
            print(f'Reusing state for {self.dir_path!r}.')
        else:
            if not os.path.exists(self.pickle_path):
                # Convert vertical to a DataFrame. It is faster than reading
                # from files (especially in the cloud where the files are
                # usually in a mounted file system which can be really slow).
                pages = [
                    page
                    for website in self._iterate_websites(FileWebsite)
                    for page in website.pages
                ]
                self.df = pd.DataFrame(
                    (
                        page.to_row()
                        for page in tqdm(pages, desc=self.pickle_path)
                    ),
                    index=[page.index for page in pages]
                )
                self.df.to_pickle(self.pickle_path)
            else:
                self.df = pd.read_pickle(self.pickle_path)

        self.websites = list(self._iterate_websites(PickleWebsite))
        self.page_count = len(self.df)

    @property
    def dir_path(self):
        return f'{self.dataset.dir_path}/{self.name}'

    @property
    def pickle_path(self):
        return f'{self.dataset.dir_path}/{self.name}.pkl'

    @property
    def groundtruth_dir(self):
        return f'{self.dataset.dir_path}/groundtruth/{self.name}'

    @property
    def groundtruth_path_prefix(self):
        return f'{self.groundtruth_dir}/{self.name}'

    def _iterate_websites(self, cls: type['Website']):
        page_count = 0
        for subdir in sorted(os.listdir(self.dir_path)):
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
            warnings.warn('Some pages were not created for site ' + \
                f'{self.dir_name} ({awe.utils.to_ranges(missing)}).')

    @property
    def dir_name(self):
        return f'{self.vertical.name}-{self.name}({self.page_count})'

    @property
    def dir_path(self):
        return f'{self.vertical.dir_path}/{self.dir_name}'

    @property
    def groundtruth_path_prefix(self):
        return f'{self.vertical.groundtruth_path_prefix}-{self.name}'

    @abc.abstractmethod
    def _iterate_pages(self):
        pass

    def _iterate_groundtruth(self):
        for file in glob.glob(f'{self.groundtruth_path_prefix}-*.txt'):
            file_name = os.path.basename(file)
            groundtruth = awe.data.set.swde_labels.GroundtruthFile(
                website=self,
                file_name=file_name,
            )
            assert os.path.samefile(groundtruth.file_path, file)
            yield groundtruth

    def get_page_at(self, idx: int):
        # Hot path: try indexing directly.
        if idx < len(self.pages) and (page := self.pages[idx]).index == idx:
            return page

        # Slow path: find the page manually.
        for page in self.pages:
            if page.index == idx:
                return page
        return None

class FileWebsite(Website):
    def _iterate_pages(self):
        for file in sorted(os.listdir(f'{self.dir_path}')):
            page = FilePage.try_create(website=self, file_name=file)
            if page is not None:
                assert page.html_file_name == file
                yield page

class PickleWebsite(Website):
    def _iterate_pages(self):
        for idx in range(self.page_count):
            yield PicklePage(website=self, index=idx)

@dataclasses.dataclass(eq=False)
class Page(awe.data.set.pages.Page):
    website: Website
    _url: Optional[str] = dataclasses.field(repr=False, init=False, default=None)
    groundtruth_entries: dict[str, awe.data.set.swde_labels.GroundtruthEntry] = \
        dataclasses.field(repr=False, default=None)

    def __post_init__(self):
        self.groundtruth_entries = {
            f.label_key: f.get_entry_for(self)
            for f in self.website.groundtruth_files.values()
        }

    @property
    def suffix(self):
        return self.website.vertical.dataset.suffix

    @property
    def file_name_no_extension(self):
        return f'{self.index:04}{self.suffix or ""}'

    @property
    def dir_path(self):
        return self.website.dir_path

    def get_labels(self):
        return awe.data.set.swde_labels.PageLabels(self)

    def to_row(self):
        return {
            'url': self.url,
            'html_text': self.get_html_text(),
            'visuals': self.load_visuals().data
        }

class FilePage(Page):
    @staticmethod
    def try_create(website: Website, file_name: str):
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

class PicklePage(Page):
    @property
    def row(self):
        return self.website.vertical.df.iloc[self.index_in_vertical]

    @property
    def url(self):
        return self.row.url

    def get_html_text(self):
        return self.row.html_text

    def load_visuals(self):
        visuals = self.create_visuals()
        visuals.data = self.row.visuals
        return visuals
