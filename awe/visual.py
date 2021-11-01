import json

from awe import awe_graph


class DomData:
    """Can load visual attributes saved by `extractor.ts`."""

    def __init__(self, path: str):
        self.path = path

    @property
    def contents(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            return f.read()

    def load(self, page: awe_graph.HtmlPage):
        """Loads DOM data from JSON into `page`."""

        data = json.loads(self.contents)
        # TODO: Load from `data` to `page`.
