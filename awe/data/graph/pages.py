import abc
import dataclasses
import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import awe.data.graph.labels


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

@dataclasses.dataclass
class Vertical:
    dataset: Dataset
    name: str
    websites: list['Website'] = dataclasses.field(repr=False, default_factory=list)

    def get_all_pages(self, *, zip_websites: bool = False):
        page_lists = (w.pages for w in self.websites)
        return get_all_pages(page_lists, zip_lists=zip_websites)

@dataclasses.dataclass
class Website:
    vertical: Vertical = dataclasses.field(repr=False)

    name: str
    """Website identifier, usually domain name."""

    pages: list['Page'] = dataclasses.field(repr=False, default_factory=list)

@dataclasses.dataclass
class Page(abc.ABC):
    website: Website = dataclasses.field(repr=False)

    @property
    @abc.abstractmethod
    def html_path(self) -> str:
        """Path to HTML file stored locally, relative to `Dataset.dir_path`."""

    @property
    @abc.abstractmethod
    def url(self) -> str:
        """Original URL of the page."""

    def get_html_text(self) -> str:
        """Obtains HTML of the page as text."""

        with open(self.html_path, encoding='utf-8') as f:
            return f.read()

    @abc.abstractmethod
    def get_labels(self) -> 'awe.data.graph.labels.PageLabels':
        """Groundtruth labeling for the page."""

def get_all_pages(page_lists: list[list[Page]], *, zip_lists: bool = False):
    if zip_lists:
        return [
            page
            for pages in itertools.zip_longest(*page_lists)
            for page in pages
            if page is not None
        ]
    return [page for pages in page_lists for page in pages]
