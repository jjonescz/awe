import awe.data.parsing
import awe.data.set.pages


class Dom:
    def __init__(self, page: awe.data.set.pages.Page):
        self.page = page
        self.tree = awe.data.parsing.parse_html(page.get_html_text())
