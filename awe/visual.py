import json
from typing import Any, Callable, Optional

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

        # Check that IDs match.
        if not node.is_text:
            real_id = node.element.attrib.get('id')
            extracted_id = node_data.get('id')
            assert real_id == extracted_id, f'IDs of {node.xpath} do not ' + \
                f'match ("{real_id}" vs "{extracted_id}").'

        # Load `node_data` into `node`.
        def load_attribute(
            snake_case: str,
            camel_case: Optional[str] = None,
            selector: Callable[[Any], Any] = lambda x: x
        ):
            val = node_data.get(camel_case or snake_case)
            if val is not None:
                setattr(node, snake_case, selector(val))

        load_attribute('box',
            selector=lambda b: awe_graph.BoundingBox(b[0], b[1], b[2], b[3]))
        load_attribute('font_family', 'fontFamily')
        load_attribute('font_size', 'fontSize')
        return True

    def find(self, xpath: str):
        elements = xpath.split('/')[1:]
        current_data = self.data
        for index, element in enumerate(elements):
            current_data = current_data.get(f'/{element}')
            if current_data is None:
                current_xpath = '/'.join(elements[:index + 1])
                raise RuntimeError(
                    f'Cannot find visual attributes for /{current_xpath}')
        return current_data
