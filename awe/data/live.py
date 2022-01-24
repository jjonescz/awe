from dataclasses import dataclass
from typing import Optional

import requests

from awe import awe_graph, utils


@dataclass
class Page(awe_graph.HtmlPage):
    url: str
    _html: Optional[str] = utils.cache_field()

    @property
    def identifier(self):
        return self.url

    @property
    def relative_original_path(self):
        return self.url

    @property
    def group_key(self):
        return self.identifier

    @property
    def group_index(self):
        return 0

    @property
    def html(self):
        if self._html is None:
            self._html = self._download_html()
        return self._html

    @property
    def labels(self):
        return NO_LABELS

    @property
    def fields(self):
        return []

    def count_label(self, _: str):
        return 0

    def _download_html(self):
        return requests.get(self.url).text

class PageLabels(awe_graph.HtmlLabels):
    def get_labels(self, _):
        return []

    def get_nodes(self, *_):
        return []

NO_LABELS = PageLabels()
