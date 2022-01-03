import colorsys
import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from awe import awe_graph, utils

if TYPE_CHECKING:
    from awe import features

XPATH_ELEMENT_REGEX = r'^/(.*?)(\[\d+\])?$'

@dataclass
class Color:
    red: int
    green: int
    blue: int
    alpha: int

    @property
    def hls(self):
        return colorsys.rgb_to_hls(self.red, self.green, self.blue)

    @property
    def hue(self):
        return self.hls[0]

    @classmethod
    def parse(cls, s: str):
        def h(i: int):
            return int(s[i:(i + 2)], 16)
        return Color(h(1), h(3), h(5), h(7))

def get_tag_name(xpath_element: str):
    return re.match(XPATH_ELEMENT_REGEX, xpath_element).group(1)

class DomData:
    """Can load visual attributes saved by `extractor.ts`."""

    data: dict[str, Any]

    def __init__(self, path: str):
        self.path = path
        self.data = {}

    @property
    def exists(self):
        return os.path.exists(self.path)

    @property
    def contents(self):
        with open(self.path, mode='r', encoding='utf-8') as file:
            return file.read()

    def read(self):
        """Reads DOM data from JSON."""
        self.data = json.loads(self.contents)

    def load_all(self, ctx: 'features.PageContextBase'):
        for node in ctx.nodes:
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
                    f'{get_xpath(tag_name, parent)} in {self.path}')

            # Add children to queue.
            for child_name, child_data in node_data.items():
                if (
                    child_name.startswith('/') and
                    ctx.node_predicate.include_visual(child_data, child_name)
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
                f'match ("{real_id}" vs "{extracted_id}") in {self.path}.'

        # Load `node_data` into `node`.
        def load_attribute(
            snake_case: str,
            selector: Callable[[Any], Any] = lambda x: x,
            default: Optional[Any] = None
        ):
            camel_case = utils.to_camel_case(snake_case)
            val = node_data.get(camel_case) or default
            if val is not None:
                try:
                    result = selector(val)
                except ValueError as e:
                    print(f'Cannot parse {snake_case}="{val}", using ' + \
                        f'default="{val}" in {self.path}: {str(e)}')
                    result = default
                setattr(node, snake_case, result)

        load_attribute('box',
            selector=lambda b: awe_graph.BoundingBox(b[0], b[1], b[2], b[3]))

        # Load visual attributes except for text fragments (they don't have
        # their own but inherit them from their container node instead).
        if not node.is_text:
            load_attribute('font_family', default='"Times New Roman"')
            load_attribute('font_size', default=16)
            load_attribute('font_weight', int, default='400')
            load_attribute('font_style', default='normal')
            load_attribute('text_align', default='start')
            load_attribute('color', Color.parse, default='#000000ff')
            load_attribute('cursor', default='auto')
            load_attribute('letter_spacing', default=0)
            load_attribute('line_height', default=node.font_size * 1.2)
            load_attribute('opacity', default=1)
            load_attribute('overflow', default='auto')
            load_attribute('pointer_events', default='auto')
            load_attribute('text_overflow', default='clip')
            load_attribute('text_transform', default='none')
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
                    f'while searching for {xpath} in {self.path}')
        return current_data
