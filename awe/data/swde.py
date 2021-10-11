import os
import re
from dataclasses import dataclass, field

import parsel

from . import constants

URL = 'https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip'
ZIP = f'{constants.DATA_DIR}/swde.zip'
DIR = f'{constants.DATA_DIR}/swde'
DATA_DIR = f'{DIR}/data'

WEBSITE_REGEX = r'^(\w+)-(\w+)\((\d+)\)$'
PAGE_REGEX = r'^(\d{4})\.htm$'
BASE_TAG_REGEX = r'^<base href="([^\n]*)"/>\w*\n(.*)'

def ignore_field(**kwargs):
    return field(init=False, repr=False, hash=False, compare=False, **kwargs)

@dataclass
class Vertical:
    name: str
    _websites: list['Website'] = ignore_field(default=None)

    @property
    def websites(self):
        if self._websites is None:
            self._websites = list(self._iterate_websites())
        return self._websites

    def _iterate_websites(self):
        for subdir in os.listdir(self.dir_path):
            website = Website(self, subdir)
            assert website.dir_name == subdir
            yield website

    @property
    def dir_path(self):
        return f'{DATA_DIR}/{self.name}'

@dataclass
class Website:
    vertical: Vertical
    name: str
    page_count: int
    _pages: list['Page'] = ignore_field(default=None)

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
        for file in os.listdir(f'{self.dir_path}'):
            page = Page(self, file)
            assert page.file_name == file
            yield page

    @property
    def dir_name(self):
        return f'{self.vertical.name}-{self.name}({self.page_count})'

    @property
    def dir_path(self):
        return f'{self.vertical.dir_path}/{self.dir_name}'

@dataclass
class Page:
    site: Website
    index: int
    _url: str = ignore_field(default=None)
    _parsed: parsel.Selector = ignore_field(default=None)

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
        self._parsed = parsel.Selector(match.group(2))

    @property
    def url(self):
        if self._url is None:
            self.parse()
        return self._url

    @property
    def parsed(self):
        if self._parsed is None:
            self.parse()
        return self._parsed

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
