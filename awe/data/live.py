from dataclasses import dataclass
from typing import Optional

import parsel
import requests

from awe import awe_graph, html_utils, utils


@dataclass
class Page(awe_graph.HtmlPage):
    url: str
    _dom: Optional[parsel.Selector] = utils.cache_field()

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
    def dom(self):
        if self._dom is None:
            self._dom = self._download_dom()
        return self._dom

    @property
    def labels(self):
        return NO_LABELS

    @property
    def fields(self):
        return []

    def count_label(self, _: str):
        return 0

    def _download_dom(self):
        text = requests.get(self.url).text
        return html_utils.parse_html(text)

class PageLabels(awe_graph.HtmlLabels):
    def get_labels(self, _):
        return []

    def get_nodes(self, *_):
        return []

NO_LABELS = PageLabels()
