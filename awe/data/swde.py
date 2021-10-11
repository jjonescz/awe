import glob
import os
import re
from dataclasses import dataclass, field

import parsel

from . import constants

URL = 'https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip'
ZIP = f'{constants.DATA_DIR}/swde.zip'
DIR = f'{constants.DATA_DIR}/swde'
DATA_DIR = f'{DIR}/data'
GROUND_TRUTH = 'groundtruth'

WEBSITE_REGEX = r'^(\w+)-(\w+)\((\d+)\)$'
PAGE_REGEX = r'^(\d{4})\.htm$'
BASE_TAG_REGEX = r'^<base href="([^\n]*)"/>\w*\n(.*)'
GROUNDTRUTH_REGEX = r'^(\w+)-(\w+)-(\w+)\.txt$'

def add_field(**kwargs):
    return field(hash=False, compare=False, **kwargs)

def ignore_field(**kwargs):
    return add_field(init=False, repr=False, **kwargs)

def cache_field(**kwargs):
    return ignore_field(default=None, **kwargs)

@dataclass
class Vertical:
    name: str
    _websites: list['Website'] = cache_field()

    @property
    def websites(self):
        if self._websites is None:
            self._websites = list(self._iterate_websites())
        return self._websites

    def _iterate_websites(self):
        for subdir in sorted(os.listdir(self.dir_path)):
            website = Website(self, subdir)
            assert website.dir_name == subdir
            yield website

    @property
    def dir_path(self):
        return f'{DATA_DIR}/{self.name}'

    @property
    def groundtruth_dir(self):
        return f'{DATA_DIR}/{GROUND_TRUTH}/{self.name}'

    @property
    def groundtruth_path_prefix(self):
        return f'{self.groundtruth_dir}/{self.name}'

@dataclass
class GroundTruthField:
    site: 'Website'
    name: str
    _entries: list['GroundTruthEntry'] = cache_field()

    def __init__(self, site: 'Website', file_name: str):
        self.site = site
        match = re.search(GROUNDTRUTH_REGEX, file_name)
        assert match.group(1) == site.vertical.name
        assert match.group(2) == site.name
        self.name = match.group(3)

    @property
    def file_name(self):
        return f'{self.site.vertical.name}-{self.site.name}-{self.name}.txt'

    @property
    def file_path(self):
        return f'{self.site.vertical.groundtruth_dir}/{self.file_name}'
        
    @property
    def lines(self):
        with open(self.file_path, mode='r', encoding='utf-8-sig') as file:
            return [line.rstrip('\r\n') for line in file]

    @property
    def entries(self):
        if self._entries is None:
            self._entries = list(self._iterate_entries())
        return self._entries

    def _iterate_entries(self):
        lines = self.lines

        # Read first line.
        vertical, site, name = lines[0].split('\t')
        assert vertical == self.site.vertical.name
        assert site == self.site.name
        assert name == self.name

        # Read second line.
        count, _, _, _ = lines[1].split('\t')
        assert int(count) == self.site.page_count

        # Read rest of the file.
        for index, line in enumerate(lines[2:]):
            expected_index, expected_nonnull_count, *values = line.split('\t')
            assert int(expected_index) == index
            parsed_values = [] if values == ['<NULL>'] else values
            assert int(expected_nonnull_count) == len(parsed_values)
            page = self.site.pages[index]
            assert page.index == index
            yield GroundTruthEntry(self, page, parsed_values)

@dataclass
class Website:
    vertical: Vertical
    name: str
    page_count: int
    _pages: list['Page'] = cache_field()
    _groundtruth: list[GroundTruthField] = cache_field()

    def __init__(self, vertical: Vertical, dir_name: str):
        match = re.search(WEBSITE_REGEX, dir_name)
        assert vertical.name == match.group(1)
        self.vertical = vertical
        self.name = match.group(2)
        self.page_count = int(match.group(3))

    @property
    def pages(self):
        if self._pages is None:
            self._pages = list(self._iterate_pages())
            assert len(self._pages) == self.page_count
        return self._pages

    def _iterate_pages(self):
        for file in sorted(os.listdir(f'{self.dir_path}')):
            page = Page(self, file)
            assert page.file_name == file
            yield page

    @property
    def dir_name(self):
        return f'{self.vertical.name}-{self.name}({self.page_count})'

    @property
    def dir_path(self):
        return f'{self.vertical.dir_path}/{self.dir_name}'

    @property
    def groundtruth_path_prefix(self):
        return f'{self.vertical.groundtruth_path_prefix}-{self.name}'

    @property
    def groundtruth(self):
        if self._groundtruth is None:
            self._groundtruth = list(self._iterate_groundtruth())
        return self._groundtruth

    def _iterate_groundtruth(self):
        for file in glob.glob(f'{self.groundtruth_path_prefix}-*.txt'):
            file_name = os.path.basename(file)
            groundtruth_field = GroundTruthField(self, file_name)
            assert groundtruth_field.file_path == file
            yield groundtruth_field

@dataclass
class Page:
    site: Website
    index: int
    _url: str = cache_field()
    _html: parsel.Selector = cache_field()

    def __init__(self, site: Website, file_name: str):
        match = re.search(PAGE_REGEX, file_name)
        self.site = site
        self.index = int(match.group(1))

    @property
    def file_name(self):
        return f'{self.index:04}.htm'

    @property
    def file_path(self):
        return f'{self.site.dir_path}/{self.file_name}'

    @property
    def contents(self):
        with open(self.file_path, mode='r', encoding='utf-8-sig') as file:
            return file.read()

    def parse(self):
        match = re.search(BASE_TAG_REGEX, self.contents, flags=re.RegexFlag.S)
        # Note that there is a `<base />` tag appended before each HTML document
        # in SWDE with the actual crawled URL.
        self._url = match.group(1)
        self._html = parsel.Selector(match.group(2))

    @property
    def url(self):
        if self._url is None:
            self.parse()
        return self._url

    @property
    def html(self):
        if self._html is None:
            self.parse()
        return self._html

@dataclass
class GroundTruthEntry:
    field: GroundTruthField
    page: Page
    values: list[str] = add_field()
    _nodes: list[parsel.Selector] = cache_field()

    @property
    def nodes(self):
        """Returns nodes from `page.html` matching the groundtruth `values`."""
        if self._nodes is None:
            self._nodes = list(self._iterate_nodes())
        return self._nodes

    def _iterate_nodes(self):
        for value in self.values:
            match = self.page.html.xpath(
                '//*[normalize-space(text()) = $value]',
                value=value
            )
            assert len(match) > 0
            yield from match

VERTICALS = [
    Vertical('auto'),
    Vertical('book'),
    Vertical('camera'),
    Vertical('job'),
    Vertical('movie'),
    Vertical('nbaplayer'),
    Vertical('restaurant'),
    Vertical('university')
]
