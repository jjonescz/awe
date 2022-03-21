import dataclasses

import awe.data.set.pages
import awe.data.set.labels


@dataclasses.dataclass(eq=False)
class Page(awe.data.set.pages.Page):
    def __init__(self, index: int, url: str, html_text: str, visuals_data: dict[str]):
        super().__init__(website=None, index=index)
        self._url = url
        self.html_text = html_text
        self.visuals_data = visuals_data

    @property
    def file_name_no_extension(self):
        return self.url

    @property
    def dir_path(self):
        return '/LIVE'

    @property
    def url(self):
        return self._url

    @property
    def index_in_dataset(self):
        return self.index

    def load_visuals(self):
        visuals = self.create_visuals()
        visuals.data = self.visuals_data
        return visuals

    def get_html_text(self):
        return self.html_text

    def get_labels(self):
        return PageLabels(self)

class PageLabels(awe.data.set.labels.PageLabels):
    """Empty labels."""

    page: Page

    @property
    def label_keys(self):
        return []

    def get_label_values(self, _: str):
        return []

    def get_labeled_nodes(self, _: str):
        return []
