import glob
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from tqdm.auto import tqdm

from awe import awe_graph, features, html_utils, utils, visual
from awe.data import constants

URL = 'https://web.archive.org/web/20210630013015id_/' + \
'https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip'
ZIP = f'{constants.DATA_DIR}/swde.zip'
DIR = f'{constants.DATA_DIR}/swde'
DATA_DIR = f'{DIR}/data'
GROUND_TRUTH = 'groundtruth'

NBSP = html_utils.unescape('&nbsp;')

WEBSITE_REGEX = r'^(\w+)-(\w+)\((\d+)\)$'
PAGE_REGEX = r'^(\d{4})(-.*)?\.htm$'
BASE_TAG_REGEX = r'^<base href="([^\n]*)"/>\w*\n(.*)'
GROUNDTRUTH_REGEX = r'^(\w+)-(\w+)-(\w+)\.txt$'

WHITESPACE_REGEX = r'([^\S\r\n]|[\u200b])'
"""Matches whitespace except newline."""

@dataclass
class Vertical:
    dataset: 'Dataset'
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
            assert page is not None, \
                f'No page at {index} in {self.site.dir_name}.'
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
            self._pages = self._get_pages()
            assert len(self._pages) == self.page_count, 'Page count ' + \
                f'inconsistent (found {len(self._pages)}, expected ' + \
                f'{self.page_count}).'
        return self._pages

    def _get_pages(self) -> list['Page']:
        result: list[Optional[Page]] = [None] * self.page_count
        for file in sorted(os.listdir(f'{self.dir_path}')):
            page = Page.try_create(self, file)
            if page is not None:
                assert page.file_name == file, 'Page name inconsistent ' + \
                    f'(computed {page.file_name}, actual {file}).'
                assert result[page.index] is None, \
                    f'Page already loaded ({page.index}).'
                result[page.index] = page

        # Verify all pages were created.
        non_existent = [str(i) for i, p in enumerate(result) if p is None]
        assert len(non_existent) == 0, f'Some pages were not created for ' + \
            f'site {self.dir_name} ({", ".join(non_existent)}).'

        return result

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
            assert os.path.samefile(groundtruth_field.file_path, file)
            yield groundtruth_field

    def find_groundtruth(self, name: str):
        for field in self.groundtruth:
            if field.name == name:
                return field
        return None

@dataclass
class Page(awe_graph.HtmlPage):
    site: Website
    index: int
    suffix: Optional[str]

    @classmethod
    def try_create(cls, site: Website, file_name: str, no_suffix: bool = False):
        match = re.search(PAGE_REGEX, file_name)
        suffix = None if no_suffix else site.vertical.dataset.suffix
        if match is None or match.group(2) != suffix:
            return None
        return Page(site, int(match.group(1)), suffix)

    @property
    def file_name(self):
        return f'{self.index:04}{self.suffix or ""}.htm'

    @property
    def file_path(self):
        return f'{self.site.dir_path}/{self.file_name}'

    @property
    def identifier(self):
        return f'{self.site.name}/{self.file_name}'

    @property
    def group_key(self):
        return f'{self.site.dir_path}/{self.suffix or ""}'

    @property
    def group_index(self):
        return self.index

    @property
    def data_point_path(self):
        return self.file_path.removesuffix('.htm') + '.pt'

    @property
    def dom_data(self):
        json_path = self.file_path.removesuffix('.htm') + '.json'
        return visual.DomData(json_path)

    @property
    def has_dom_data(self):
        return self.dom_data.exists

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
        return html_utils.parse_html(self.html)

    @property
    def labels(self):
        return PageLabels(self)

    @property
    def fields(self):
        return [field.name for field in self.site.groundtruth]

    def count_label(self, label: str):
        return len(self.get_groundtruth_texts(label))

    def get_groundtruth_texts(self, label: str):
        field = self.site.find_groundtruth(label)
        if field is None:
            return []
        entry = field.entries[self.index]
        assert entry.page == self
        return entry.values

    def prepare(self, ctx: features.PageContext):
        try:
            dom_data = self.dom_data
            dom_data.read()
            dom_data.load_all(ctx)
        except Exception as e:
            raise RuntimeError(f'Cannot prepare page {self.file_path}') from e

class PageLabels(awe_graph.HtmlLabels):
    nodes: dict[str, list[str]]
    """Map label -> groundtruth XPaths."""

    def __init__(self, page: Page):
        self.page = page
        self.nodes = {}
        for groundtruth_field in page.site.groundtruth:
            entry = groundtruth_field.entries[page.index]
            assert entry.page == page
            self.nodes[groundtruth_field.name] = entry.nodes

    def get_labels(self, node: awe_graph.HtmlNode):
        return list(self._iter_labels(node))

    def get_nodes(self, label: str):
        xpaths = self.nodes[label]
        return [
            node
            for node in self.page.initialize_tree().descendants
            if node.xpath_swde in xpaths
        ]

    def _iter_labels(self, node: awe_graph.HtmlNode):
        for label, xpaths in self.nodes.items():
            if node.xpath_swde in xpaths:
                yield label

@dataclass
class GroundTruthEntry:
    field: GroundTruthField
    page: Page
    values: list[str] = utils.add_field()
    """Values for the field as loaded from the groundtruth file."""

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
        # characters before matching. See AWE-1.
        page_html = re.sub(WHITESPACE_REGEX, ' ', page_html)

        page_dom = html_utils.parse_html(page_html)

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

class Dataset:
    def __init__(self, suffix: Optional[str] = None):
        self.suffix = suffix
        self.verticals = [Vertical(self, name) for name in VERTICAL_NAMES]

    def validate(self, *,
        pages: Optional[list[Page]] = None,
        verticals: Optional[list[Vertical]] = None,
        parallelize: Optional[int] = None,
        skip: int = 0,
        collect_errors: bool = False,
        error_callback: Optional[Callable[[int, Page, AssertionError], None]] = None
    ):
        def validate_one(t: tuple[int, Page]):
            index, page = t
            try:
                _ = page.labels
                return None
            except AssertionError as e:
                if error_callback is not None:
                    error_callback(index, page, e)
                if collect_errors:
                    return index, e
                else:
                    raise RuntimeError(
                        f'Cannot validate page at {index}') from e

        if pages is None:
            pages = [
                p
                for v in verticals or self.verticals
                for w in v.websites
                for p in w.pages
            ]

        target_pages = list(enumerate(pages))[skip:]

        results = utils.parallelize(
            parallelize,
            validate_one,
            target_pages,
            desc='pages'
        )

        if collect_errors:
            def transform(r: tuple[int, AssertionError]):
                index, e = r
                page = pages[index]
                return index, page, e
            return [transform(r) for r in results if r is not None]
