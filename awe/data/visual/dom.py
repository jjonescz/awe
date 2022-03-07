import json
import os
import warnings
from typing import Any, Callable

import awe.data.graph.dom
import awe.data.visual.attribute
import awe.data.visual.structs
import awe.utils


class DomData:
    """Can load visual attributes saved by `extractor.ts`."""

    data: dict[str, Any] = None

    def __init__(self, path: str):
        self.path = path

    def exists(self):
        return os.path.exists(self.path)

    def get_json_str(self):
        with open(self.path, mode='r', encoding='utf-8') as file:
            return file.read()

    def load_json(self):
        """Reads DOM data from JSON."""
        self.data = json.loads(self.get_json_str())

    def fill_tree(self, dom: awe.data.graph.dom.Dom):
        for node in dom.nodes:
            self.fill_one(node)

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

            # Check `fill_one` was called on this entry.
            filled = node_data.pop('_filled', False)
            if not filled and tag_name != '':
                raise RuntimeError('Unused visual attributes for ' + \
                    f'{get_xpath(tag_name, parent)!r} in {self.path!r}.')

            # Add children to queue.
            for child_name, child_data in node_data.items():
                if child_name.startswith('/'):
                    queue.insert(0, (child_data, child_name, item))

    def fill_one(self, node: awe.data.graph.dom.Node):
        xpath = node.get_xpath()
        node_data = self.find(xpath)
        node_data['_filled'] = True

        # Check that IDs match.
        if not node.is_text:
            real_id = node.id
            extracted_id = node_data.get('id')
            assert real_id == extracted_id, f'IDs of {xpath!r} do not ' + \
                f'match ({real_id=} vs {extracted_id=}) in {self.path!r}.'

        # Load `node_data` into `node`.
        def load_attribute(
            snake_case: str,
            parser: Callable[[Any, dict[str, Any]], Any] = lambda x: x,
            default: Callable[[awe.data.graph.dom.Node], Any] = lambda _: None
        ):
            camel_case = awe.utils.to_camel_case(snake_case)
            val = node_data.get(camel_case) or default(node)
            if val is not None:
                try:
                    result = parser(val, node_data)
                except ValueError as e:
                    warnings.warn(f'Cannot parse {snake_case}={val!r} ' + \
                        f'using default={val!r} in {self.path!r}: {str(e)}')
                    result = default(node)
                return result
            return None

        node.box = load_attribute('box', parser=lambda b, _: \
            awe.data.visual.structs.BoundingBox(b[0], b[1], b[2], b[3]))

        # Load visual attributes except for text fragments (they don't have
        # their own but inherit them from their container node instead).
        if not node.is_text:
            for a in awe.data.visual.attribute.VISUAL_ATTRIBUTES.values():
                node.visuals[a.name] = load_attribute(
                    a.name, a.parse, a.get_default)
        return True

    def find(self, xpath: str):
        elements = xpath.split('/')[1:]
        current_data = self.data
        for index, element in enumerate(elements):
            current_data = current_data.get(f'/{element}')
            if current_data is None:
                current_xpath = '/' + '/'.join(elements[:index + 1])
                raise RuntimeError(
                    f'Cannot find visual attributes for {current_xpath!r} ' + \
                    f'while searching for {xpath!r} in {self.path!r}.')
        return current_data
