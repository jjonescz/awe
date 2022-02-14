import abc
import dataclasses


@dataclasses.dataclass
class Dataset:
    name: str

    dir_path: str = dataclasses.field(repr=False)
    """Path to root directory of the dataset, relative to project root dir."""

    verticals: list['Vertical'] = dataclasses.field(repr=False, default_factory=list)

@dataclasses.dataclass
class Vertical:
    dataset: Dataset
    name: str
    websites: list['Website'] = dataclasses.field(repr=False, default_factory=list)

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
