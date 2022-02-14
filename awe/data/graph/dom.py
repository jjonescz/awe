import awe.data.graph.pages
import awe.data.parsing

class Dom:
    def __init__(self, page: awe.data.graph.pages.Page):
        self.page = page
        self.tree = awe.data.parsing.parse_html(page.get_html_text())
