import dataclasses
import glob
import os
import re
import warnings
from typing import Optional

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

    def __init__(self, suffix: Optional[str] = None):
        super().__init__(
            name=f'swde{suffix or ""}',
            dir_path=DATA_DIR,
        )
        self.suffix = suffix
        self.verticals = [
            Vertical(dataset=self, name=name)
            for name in VERTICAL_NAMES
        ]

@dataclasses.dataclass
class Vertical(awe.data.set.pages.Vertical):
    dataset: Dataset
    websites: list['Website'] = dataclasses.field(repr=False, default_factory=list)

    def __post_init__(self):
        self.websites = list(self._iterate_websites())

    @property
    def dir_path(self):
        return f'{self.dataset.dir_path}/{self.name}'

    @property
    def groundtruth_dir(self):
        return f'{self.dataset.dir_path}/groundtruth/{self.name}'

    @property
    def groundtruth_path_prefix(self):
        return f'{self.groundtruth_dir}/{self.name}'

    def _iterate_websites(self):
        if not os.path.exists(self.dir_path):
            warnings.warn(
                f'Vertical directory does not exist ({self.dir_path}).')
            return

        for subdir in sorted(os.listdir(self.dir_path)):
            website = Website(self, subdir)
            assert website.dir_name == subdir

            # HACK: Skip website careerbuilder.com whose groundtruth values are
            # in HTML comments (that's bug in the dataset).
            if self.name == 'job' and website.name == 'careerbuilder':
                continue

            yield website

@dataclasses.dataclass
class Website(awe.data.set.pages.Website):
    vertical: Vertical
    pages: list['Page'] = dataclasses.field(repr=False)
    page_count: int = None
    groundtruth_files: dict[str, awe.data.set.swde_labels.GroundtruthFile] = \
        dataclasses.field(repr=False, default=None)

    def __init__(self, vertical: Vertical, dir_name: str):
        match = re.search(WEBSITE_REGEX, dir_name)
        assert vertical.name == match.group(1)
        super().__init__(
            vertical=vertical,
            name=match.group(2)
        )
        self.page_count = int(match.group(3))
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

    def _iterate_pages(self):
        for file in sorted(os.listdir(f'{self.dir_path}')):
            page = Page.try_create(website=self, file_name=file)
            if page is not None:
                assert page.html_file_name == file
                yield page

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

@dataclasses.dataclass
class Page(awe.data.set.pages.Page):
    website: Website
    index: int = None
    _url: Optional[str] = dataclasses.field(repr=False, init=False, default=None)
    groundtruth_entries: dict[str, awe.data.set.swde_labels.GroundtruthEntry] = \
        dataclasses.field(repr=False, default=None)

    @staticmethod
    def try_create(website: Website, file_name: str):
        match = re.search(PAGE_REGEX, file_name)
        if match is None:
            return None
        page = Page(website=website, index=int(match.group(1)))
        suffix = match.group(2)
        if suffix != page.suffix:
            return None

        page.groundtruth_entries = {
            f.label_key: f.get_entry_for(page)
            for f in website.groundtruth_files.values()
        }

        return page

    @property
    def suffix(self):
        return self.website.vertical.dataset.suffix

    @property
    def html_file_name(self):
        return f'{self.index:04}{self.suffix or ""}.htm'

    @property
    def html_path(self):
        return f'{self.website.dir_path}/{self.html_file_name}'

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

    def get_labels(self):
        return awe.data.set.swde_labels.PageLabels(self)
