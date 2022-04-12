import abc
import collections
import dataclasses
import gc
import itertools
import urllib.parse
import os
from typing import TYPE_CHECKING, Optional

from tqdm.auto import tqdm

import awe.data.graph.dom
import awe.data.visual.dom

if TYPE_CHECKING:
    import awe.data.set.labels


@dataclasses.dataclass
class ClearCacheRequest:
    dom: bool = True
    labels: bool = True
    dom_dirty_flags: bool = False

@dataclasses.dataclass
class Dataset:
    name: str

    dir_path: str = dataclasses.field(repr=False)
    """Path to root directory of the dataset, relative to project root dir."""

    verticals: list['Vertical'] = dataclasses.field(repr=False, default_factory=list)

    def get_all_pages(self, *, zip_verticals: bool = False, zip_websites: bool = False):
        page_lists = (
            v.get_all_pages(zip_websites=zip_websites)
            for v in self.verticals
        )
        return get_all_pages(page_lists, zip_lists=zip_verticals)

    def clear_cache(self, request: ClearCacheRequest):
        for page in self.get_all_pages():
            page.clear_cache(request)
        return gc.collect()

@dataclasses.dataclass
class Vertical:
    dataset: Dataset
    name: str
    websites: list['Website'] = dataclasses.field(repr=False, default_factory=list)
    prev_page_count: int = dataclasses.field(repr=False, default=None)
    page_count: int = dataclasses.field(repr=False, default=None)

    def get_all_pages(self, *, zip_websites: bool = False):
        page_lists = (w.pages for w in self.websites)
        return get_all_pages(page_lists, zip_lists=zip_websites)

@dataclasses.dataclass
class Website:
    vertical: Vertical = dataclasses.field(repr=False)

    name: str
    """Website identifier, usually domain name."""

    pages: list['Page'] = dataclasses.field(repr=False, default_factory=list)

    prev_page_count: int = dataclasses.field(repr=False, default=None)
    page_count: int = dataclasses.field(repr=False, default=None)

    @property
    @abc.abstractmethod
    def variable_nodes_file_path(self) -> str:
        """Path to a file containing list of variable node XPaths."""

    def get_common_prefix(self):
        return os.path.commonprefix([p.url for p in self.pages])

    def get_domain(self):
        return urllib.parse.urlparse(self.get_common_prefix()).netloc

    def find_variable_xpaths(self,
        max_variable_nodes_per_website: int = 300,
    ) -> set[str]:
        """
        Determines whether text nodes are variable or fixed in pages across the
        website (as defined in the SimpDOM paper).
        Returns the set of variable XPaths.
        """

        # Try cache first.
        if os.path.exists(self.variable_nodes_file_path):
            with open(self.variable_nodes_file_path, mode='r', encoding='utf-8') as f:
                return set(line.rstrip() for line in f)

        # Find texts for each XPath across pages.
        nodes = collections.defaultdict(set) # XPath -> set of texts
        for page in tqdm(self.pages, desc='texts', leave=False):
            page: Page
            dom = page.dom
            dom.init_nodes()
            for node in dom.nodes:
                if node.is_text:
                    nodes[node.get_xpath()].add(node.text)

        # Sort nodes by variability.
        node_vars = sorted(
            ((xpath, len(texts)) for xpath, texts in nodes.items()),
            key=lambda p: p[1],
            reverse=True
        )

        # Ensure labeled nodes are variable.
        labeled_xpaths = set()
        for page in tqdm(self.pages, desc='labels', leave=False):
            page: Page
            dom = page.dom
            dom.init_nodes()
            dom.init_labels(propagate_to_descendants=True)
            for labeled_groups in dom.labeled_nodes.values():
                for labeled_nodes in labeled_groups:
                    for node in labeled_nodes:
                        labeled_xpaths.add(node.get_xpath())

        # Split XPaths into variable/fixed sets.
        variable_nodes = set() # XPaths
        for xpath, variability in node_vars:
            if ((
                    variability > 5
                    and len(variable_nodes) < max_variable_nodes_per_website
                )
                or xpath in labeled_xpaths
            ):
                variable_nodes.add(xpath)

        # Save to cache.
        with open(self.variable_nodes_file_path, mode='w', encoding='utf-8') as f:
            f.writelines(f'{xpath}\n' for xpath in variable_nodes)
        print(f'Saved {self.variable_nodes_file_path!r}.')

        return variable_nodes

@dataclasses.dataclass(eq=False)
class Page(abc.ABC):
    website: Website = dataclasses.field(repr=False)
    index: int = dataclasses.field(repr=False, default=None)
    _labels = None
    _dom = None

    valid: Optional[bool] = dataclasses.field(repr=False, default=None)
    """
    Set by various components of the validation pipeline (hence the shared
    state).
    """

    @property
    def original_file_name_no_extension(self):
        return self.file_name_no_extension

    @property
    @abc.abstractmethod
    def file_name_no_extension(self) -> str:
        """Name of page stored locally."""

    @property
    @abc.abstractmethod
    def dir_path(self) -> str:
        """
        Path to directory where page files are stored locally, relative to
        `Dataset.dir_path`.
        """

    @property
    def original_html_file_name(self):
        return f'{self.original_file_name_no_extension}.htm'

    @property
    def html_file_name(self):
        return f'{self.file_name_no_extension}.htm'

    @property
    def original_html_path(self):
        """
        Path to non-modified HTML file (can be different from `html_path` if
        that points to a snapshot saved by the JavaScript scraper).
        """
        return f'{self.dir_path}/{self.original_html_file_name}'

    @property
    def html_path(self):
        return f'{self.dir_path}/{self.html_file_name}'

    @property
    def visuals_suffix(self):
        return ''

    @property
    def visuals_file_name(self):
        return f'{self.file_name_no_extension}{self.visuals_suffix}.json'

    @property
    def visuals_path(self):
        return f'{self.dir_path}/{self.visuals_file_name}'

    @property
    def screenshot_file_name(self):
        return f'{self.file_name_no_extension}{self.visuals_suffix}-full.png'

    @property
    def screenshot_path(self):
        return f'{self.dir_path}/{self.screenshot_file_name}'

    @property
    @abc.abstractmethod
    def url(self) -> str:
        """Original URL of the page."""

    @property
    def index_in_vertical(self):
        return self.website.prev_page_count + self.index

    @property
    def index_in_dataset(self):
        return self.website.vertical.prev_page_count + self.index_in_vertical

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self.get_labels()
        return self._labels

    @property
    def dom(self):
        if self._dom is None:
            return self._create_dom()
        return self._dom

    def try_get_dom(self):
        return self._dom

    def cache_dom(self):
        # Note that creating page DOM can be memory consuming (especially if
        # done for many pages), hence this method exists to make it explicit.
        if self._dom is None:
            self._dom = self._create_dom()
        return self._dom

    def _create_dom(self):
        return awe.data.graph.dom.Dom(self)

    def create_visuals(self):
        return awe.data.visual.dom.DomData(self.visuals_path)

    @abc.abstractmethod
    def load_visuals(self) -> awe.data.visual.dom.DomData:
        """Loads visual attributes for the page."""

    def clear_cache(self, request: ClearCacheRequest):
        if request.dom or request.dom_dirty_flags:
            self.website.found_variable_nodes = False
        if request.dom:
            self._dom = None
        if request.labels:
            self._labels = None
        if request.dom_dirty_flags and self._dom is not None:
            self._dom.friend_cycles_computed = False
            self._dom.visual_neighbors_computed = False

    def get_html_text(self):
        """Obtains HTML of the page as text."""

        with open(self.html_path, encoding='utf-8') as f:
            return f.read()

    @abc.abstractmethod
    def get_labels(self) -> 'awe.data.set.labels.PageLabels':
        """Groundtruth labeling for the page."""

    def __eq__(self, other: 'Page'):
        return other and self.index_in_dataset == other.index_in_dataset \
            and self.website.vertical.dataset.name == other.website.vertical.dataset.name

    def __ne__(self, other: 'Page'):
        return not self.__eq__(other)

    def __hash__(self):
        # Usually, dataset is not different so we don't bother hashing it.
        return self.index_in_dataset

def get_all_pages(page_lists: list[list[Page]], *, zip_lists: bool = False):
    if zip_lists:
        return [
            page
            for pages in itertools.zip_longest(*page_lists)
            for page in pages
            if page is not None
        ]
    return [page for pages in page_lists for page in pages]
