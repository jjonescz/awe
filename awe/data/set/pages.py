import abc
import dataclasses
import itertools
from typing import TYPE_CHECKING

import awe.data.graph.dom

if TYPE_CHECKING:
    import awe.data.set.labels


@dataclasses.dataclass
class ClearCacheRequest:
    dom: bool = True
    labels: bool = True

@dataclasses.dataclass
class Dataset:
    name: str

    dir_path: str = dataclasses.field(repr=False)
    """Path to root directory of the dataset, relative to project root dir."""

    verticals: list['Vertical'] = dataclasses.field(repr=False, default_factory=list)

    def get_state(self):
        """
        Extracts an internal state that can be later restored by passing it into
        `Dataset` constructor.
        """

    def get_all_pages(self, *, zip_verticals: bool = False, zip_websites: bool = False):
        page_lists = (
            v.get_all_pages(zip_websites=zip_websites)
            for v in self.verticals
        )
        return get_all_pages(page_lists, zip_lists=zip_verticals)

    def clear_cache(self, request: ClearCacheRequest):
        for page in self.get_all_pages():
            page.clear_cache(request)

    def clear_predictions(self):
        for page in self.get_all_pages():
            page.clear_predictions()

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

@dataclasses.dataclass(eq=False)
class Page(abc.ABC):
    website: Website = dataclasses.field(repr=False)
    index: int = dataclasses.field(repr=False, default=None)
    _labels = None
    _dom = None

    @property
    @abc.abstractmethod
    def html_path(self) -> str:
        """Path to HTML file stored locally, relative to `Dataset.dir_path`."""

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
            self._dom = awe.data.graph.dom.Dom(self)
        return self._dom

    def try_get_dom(self):
        return self._dom

    def clear_cache(self, request: ClearCacheRequest):
        if request.dom:
            self._dom = None
        if request.labels:
            self._labels = None

    def clear_predictions(self):
        if self._dom is not None:
            self._dom.clear_predictions()

    def get_html_text(self) -> str:
        """Obtains HTML of the page as text."""

        with open(self.html_path, encoding='utf-8') as f:
            return f.read()

    @abc.abstractmethod
    def get_labels(self) -> 'awe.data.set.labels.PageLabels':
        """Groundtruth labeling for the page."""

    def __eq__(self, other: 'Page'):
        return other and self.index_in_dataset == other.index_in_dataset \
            and self.website.vertical.dataset == other.website.vertical.dataset

    def __ne__(self, other: 'Page'):
        return not self.__eq__(other)

    def __hash__(self):
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
