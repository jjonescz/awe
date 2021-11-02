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
        # Ignore text fragments (only nodes are extracted).
        if node.is_text:
            return False

        node_data = self.find(node.xpath)

        # Check that IDs match.
        if not node.is_text:
            real_id = node.element.attrib.get('id')
            extracted_id = node_data.get('id')
            assert real_id == extracted_id, f'IDs of {node.xpath} do not ' + \
                f'match ("{real_id}" vs "{extracted_id}").'

        # TODO: Load `node_data` into `node`.
        return True

    def find(self, xpath: str):
        elements = xpath.split('/')[1:]
        current_data = self.data
        for index, element in enumerate(elements):
            current_data = current_data.get(f'/{element}')
            if current_data is None:
                current_xpath = '/'.join(elements[:index + 1])
                raise RuntimeError(f'Cannot find element at /{current_xpath}')
        return current_data
