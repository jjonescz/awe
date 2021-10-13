import glob
import os
import re
from dataclasses import dataclass

import parsel
from tqdm.auto import tqdm

from awe import awe_graph, html_utils, utils
from awe.data import constants

URL = 'https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip'
ZIP = f'{constants.DATA_DIR}/swde.zip'
DIR = f'{constants.DATA_DIR}/swde'
DATA_DIR = f'{DIR}/data'
GROUND_TRUTH = 'groundtruth'

NBSP = html_utils.unescape('&nbsp;')

WEBSITE_REGEX = r'^(\w+)-(\w+)\((\d+)\)$'
PAGE_REGEX = r'^(\d{4})\.htm$'
BASE_TAG_REGEX = r'^<base href="([^\n]*)"/>\w*\n(.*)'
GROUNDTRUTH_REGEX = r'^(\w+)-(\w+)-(\w+)\.txt$'

WHITESPACE_REGEX = r'([^\S\r\n]|[\u200b])'
"""Matches whitespace except newline."""

@dataclass
class Vertical:
    name: str
    _websites: list['Website'] = utils.cache_field()

    @property
    def websites(self):
        if self._websites is None:
            self._websites = list(self._iterate_websites())
        return self._websites

    def _iterate_websites(self):
        for subdir in sorted(os.listdir(self.dir_path)):
            website = Website(self, subdir)
            assert website.dir_name == subdir

            # HACK: Skip website careerbuilder.com whose groundtruth values are
            # in HTML comments (that's bug in the dataset).
            if self.name == 'job' and website.name == 'careerbuilder':
                continue

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
    _entries: list['GroundTruthEntry'] = utils.cache_field()

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
    _pages: list['Page'] = utils.cache_field()
    _groundtruth: list[GroundTruthField] = utils.cache_field()

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
class Page(awe_graph.HtmlPage):
    site: Website
    index: int

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
        url = match.group(1)
        html = match.group(2)
        return url, html

    @property
    def url(self):
        return self.parse()[0]

    @property
    def html(self):
        return self.parse()[1]

    @property
    def dom(self):
        return parsel.Selector(self.html)

    @property
    def labels(self):
        return PageLabels(self)

class PageLabels(awe_graph.HtmlLabels):
    nodes: dict[str, list[str]]
    """Map label -> groundtruth XPaths."""

    def __init__(self, page: Page):
        self.nodes = dict()
        for groundtruth_field in page.site.groundtruth:
            entry = groundtruth_field.entries[page.index]
            assert entry.page == page
            self.nodes[groundtruth_field.name] = entry.nodes

    def get_labels(self, node: awe_graph.HtmlNode):
        return list(self._iter_labels(node))

    def _iter_labels(self, node: awe_graph.HtmlNode):
        for label, xpaths in self.nodes.items():
            if node.xpath in xpaths:
                yield label

@dataclass
class GroundTruthEntry:
    field: GroundTruthField
    page: Page
    values: list[str] = utils.add_field()

    @property
    def nodes(self):
        """
        Returns XPaths to nodes from `page.html` matching the groundtruth
        `values`.
        """
        return list(self._iterate_nodes())

    def _iterate_nodes(self):
        page_html = self.page.html

        # HACK: If there are HTML-encoded spaces in the HTML (e.g., `&nbsp;`),
        # they are preserved in the groundtruth. If they're there as plain
        # Unicode characters (not HTML-encoded), they're not preserved. This is
        # a bug/inconsistency in the dataset. Therefore, we remove these
        # characters before matching.
        page_html = re.sub(WHITESPACE_REGEX, ' ', page_html)

        page_dom = parsel.Selector(page_html)

        for value in self.values:
            # Note that this XPath is written so that it finds text fragments X,
            # Y, Z separately in HTML `<p>X<br>Y<br>Z</p>`.
            args = { 'value': html_utils.unescape(value) }
            match = page_dom.xpath(
                '//text()[normalize-space(.) = $value]',
                **args
            )

            # HACK: In some groundtruth data, unbreakable spaces are ignored.
            if len(match) == 0:
                def normalize(arg):
                    return f'normalize-space(translate({arg}, "{NBSP}", " "))'
                match = page_dom.xpath(
                    f'//text()[{normalize(".")} = {normalize("$value")}]',
                    **args
                )

            # HACK: In some groundtruth data, newlines are completely ignored.
            if len(match) == 0:
                match = page_dom.xpath(
                    '//text()[normalize-space(' +
                    'translate(., "\n", "")) = $value]',
                    **args
                )

            assert len(match) > 0, \
                f'No match found for {self.field.name}="{value}" in ' + \
                f'{self.page.file_path}.'

            for node in match:
                yield html_utils.get_xpath(node, page_dom, **args)

VERTICALS = [
    Vertical('auto'),
    Vertical('book'),
    Vertical('camera'),
    Vertical('job'),
    # HACK: Skip movie vertical as there are multiple bugs in the dataset:
    # - MPAA rating is only first character (e.g., "P" in the groundtruth but
    #   "PG13" in the HTML),
    # - director is not complete (e.g., "Roy Hill" in the groundtruth but
    #   "Geogre Roy Hill" in the HTML).
    #Vertical('movie'),
    Vertical('nbaplayer'),
    Vertical('restaurant'),
    Vertical('university')
]

def validate(*, verticals_skip=0):
    for vertical in tqdm(VERTICALS[verticals_skip:], desc='verticals'):
        for website in tqdm(vertical.websites, desc='websites', leave=False):
            for groundtruth_field in tqdm(website.groundtruth, desc='fields', leave=False):
                for entry in groundtruth_field.entries:
                    _ = entry.nodes
