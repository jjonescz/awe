import json

from awe import awe_graph


class DomData:
    """Can load visual attributes saved by `extractor.ts`."""

    data = {}

    def __init__(self, path: str):
        self.path = path

    @property
    def contents(self):
        with open(self.path, mode='r', encoding='utf-8') as file:
            return file.read()

    def read(self):
        """Reads DOM data from JSON."""
        self.data = json.loads(self.contents)

    def load_all(self, nodes: list[awe_graph.HtmlNode]):
        for node in nodes:
            self.load_one(node)

    def load_one(self, node: awe_graph.HtmlNode):
        node_data = self.find(node.xpath)
        # TODO: Load `node_data` into `node`.

    def find(self, xpath: str):
        elements = xpath.split('/')
        current_data = self.data
        for element in elements:
            current_data = current_data[f'/{element}']
        return current_data
