from dataclasses import dataclass
from typing import Optional

import parsel
import requests

from awe import awe_graph, utils


@dataclass
class Page(awe_graph.HtmlPage):
    url: str
    _dom: Optional[parsel.Selector] = utils.cache_field()

    @property
    def dom(self):
        if self._dom is None:
            self._dom = self._download_dom()
        return self._dom

    @property
    def labels(self):
        return NO_LABELS

    def _download_dom(self):
        text = requests.get(self.url).text
        return parsel.Selector(text)

class PageLabels(awe_graph.HtmlLabels):
    def get_labels(self, _: awe_graph.HtmlNode):
        return []

NO_LABELS = PageLabels()
