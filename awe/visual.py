import json
import re
from typing import Any, Callable, Optional

from awe import awe_graph
from awe.data import dataset


XPATH_ELEMENT_REGEX = r'^/(.*?)(\[\d+\])?$'

def get_tag_name(xpath_element: str):
    return re.match(XPATH_ELEMENT_REGEX, xpath_element).group(1)

class DomData:
    """Can load visual attributes saved by `extractor.ts`."""

    data: dict[str, Any] = {}

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

        # Check that all extracted data were used.
        queue = [(self.data, '', None)]
        def get_xpath(tag_name: str, parent, suffix = ''):
            """Utility for reconstructing XPath in case of error."""
            xpath = f'{tag_name}{suffix}'
            if parent is not None:
                return get_xpath(parent[1], parent[2], xpath)
            return xpath
        while len(queue) > 0:
            item = queue.pop()
            node_data, tag_name, parent = item

            # Check this entry has node attached to it (so `load_one` was called
            # on it).
            node = node_data.get('_node')
            if node is None and tag_name != '':
                raise RuntimeError('Unused visual attributes for ' + \
                    f'{get_xpath(tag_name, parent)}')

            # Add children to queue.
            for child_name, child_data in node_data.items():
                if (
                    child_name.startswith('/') and
                    # Ignore filtered tags.
                    get_tag_name(child_name) not in dataset.IGNORED_TAG_NAMES
                ):
                    queue.insert(0, (child_data, child_name, item))

    def load_one(self, node: awe_graph.HtmlNode):
        node_data = self.find(node.xpath)
        node_data['_node'] = node

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
                    f'Cannot find visual attributes for /{current_xpath} ' + \
                    f'while searching for {xpath}')
        return current_data
