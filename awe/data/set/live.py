import dataclasses

import awe.data.set.pages


@dataclasses.dataclass(eq=False)
class Page(awe.data.set.pages.Page):
    def __init__(self, url: str, html: str):
        super().__init__(website=None)
        self._url = url
        self.html = html

    @property
    def file_name_no_extension(self):
        return self.url

    @property
    def dir_path(self):
        return '/LIVE/'

    @property
    def url(self):
        return self._url

    def load_visuals(self):
        pass

    def get_html_text(self):
        return self.html

    def get_labels(self):
        raise RuntimeError(f'Cannot get labels for live page {self.url!r}.')
