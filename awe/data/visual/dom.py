import json
import os
import re
import warnings
from typing import Any, Callable

import awe.data.graph.dom
import awe.data.html_utils
import awe.data.parsing
import awe.data.visual.attribute
import awe.data.visual.structs
import awe.training.params
import awe.utils

XPATH_ELEMENT_REGEX = re.compile(r'^/(.*?)(\[(\d+)\])?$')

class DomData:
    """Can load visual attributes saved by `extractor.ts`."""

    data: dict[str, Any] = None

    def __init__(self, path: str):
        self.path = path

    def exists(self):
        """Whether the visuals JSON file exists."""

        return os.path.exists(self.path)

    def get_json_str(self):
        """Reads the visuals JSON stored on disk into a string."""

        with open(self.path, mode='r', encoding='utf-8') as file:
            return file.read()

    def load_json_str(self, json_text: str):
        """Loads DOM data from a JSON string."""

        self.data = json.loads(json_text)

    def load_json(self):
        """Reads DOM data from the JSON stored on disk."""

        self.load_json_str(self.get_json_str())

    def fill_tree_light(self,
        dom: awe.data.graph.dom.Dom,
        attrs: list[awe.data.visual.attribute.VisualAttribute] = (),
    ):
        """
        Lighter version of `fill_tree` that loads only bounding boxes (for all
        nodes) and visuals specified in the parameter `attrs` (for nodes with
        `needs_visuals` set), without any validation.
        """

        # Find the root element.
        if (root := self.data.get(f'/{dom.root.html_tag}')) is None:
            warnings.warn(
                f'Cannot find {dom.root.html_tag!r} in data ' +
                f'{self.data.keys()!r} ({dom.page.html_path!r}).')
            return

        queue = [(dom.root, root)]
        while len(queue) != 0:
            node, data = queue.pop()
            data: dict[str]

            # Load node's visuals.
            if (box := data.get('box')) is not None:
                node.box = awe.data.visual.structs.BoundingBox(*box)
            if node.needs_visuals:
                for attr in attrs:
                    self.load_visual_attribute(data, node, attr)

            # Add children to queue.
            for child in node.children:
                # Find visuals corresponding to the child.
                xpath_element = child.get_xpath_element()
                child_data = data.get(f'/{xpath_element}', None)
                if child_data is None:
                    warnings.warn(
                        f'Cannot find {xpath_element!r} in ' +
                        f'{node.get_xpath()!r} ({dom.page.html_path!r}).'
                    )
                else:
                    queue.append((child, child_data))

    def fill_tree(self, dom: awe.data.graph.dom.Dom):
        """
        Loads all visuals and validates that the visuals DOM data correspond to
        the HTML DOM tree.
        """

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
                raise RuntimeError('Unused visual attributes for ' +
                    f'{get_xpath(tag_name, parent)!r} in {self.path!r}.')

            # Add children to queue.
            for child_name, child_data in node_data.items():
                if child_name.startswith('/') and (
                    get_tag_name(child_name)
                    not in awe.data.parsing.IGNORED_TAG_NAMES
                ):
                    queue.insert(0, (child_data, child_name, item))

    def fill_one(self, node: awe.data.graph.dom.Node):
        """Loads visuals for one `node`."""

        xpath = node.get_xpath()
        node_data = self.find(xpath)
        node_data['_filled'] = True

        # Check that IDs match.
        if not node.is_text:
            real_id = node.id
            extracted_id = node_data.get('id')
            assert real_id == extracted_id, f'IDs of {xpath!r} do not ' + \
                f'match ({real_id=} vs {extracted_id=}) in {self.path!r}.'

        # Load bounding box.
        node.box = self.load_attribute(node_data, node, 'box',
            parser=lambda b, _: awe.data.visual.structs.BoundingBox(
                b[0], b[1], b[2], b[3]
            )
        )

        # Load visual attributes except for text fragments (they don't have
        # their own but inherit them from their container node instead).
        if not node.is_text:
            for a in awe.data.visual.attribute.VISUAL_ATTRIBUTES.values():
                self.load_visual_attribute(node_data, node, a)
        return True

    def load_attribute(self,
        node_data: dict[str],
        node: awe.data.graph.dom.Node,
        snake_case: str,
        parser: Callable[[Any, dict[str, Any]], Any] = lambda x: x,
        default: Callable[[awe.data.graph.dom.Node], Any] = lambda _: None
    ):
        """
        Loads a visual attribute from `node_data` element (from visuals JSON) of
        a `node`.
        """

        camel_case = awe.utils.to_camel_case(snake_case)
        val = node_data.get(camel_case) or default(node)
        if val is not None:
            try:
                result = parser(val, node_data)
            except ValueError as e:
                d = default(node)
                warnings.warn(f'Cannot parse {snake_case}={val!r} ' +
                    f'using default={d!r} in {self.path!r}: {str(e)}')
                node.dom.page.valid = False
                result = parser(d, node_data)
            return result
        return None

    def load_visual_attribute(self,
        node_data: dict[str],
        node: awe.data.graph.dom.Node,
        attr: awe.data.visual.attribute.VisualAttribute,
    ):
        """Loads visual `attr` into `node.visuals`."""

        node.visuals[attr.name] = self.load_attribute(
            node_data, node, attr.name, attr.parse, attr.get_default
        )

    def find(self, xpath: str):
        """Finds node data element in visuals DOM data given `xpath`."""

        elements = xpath.split('/')[1:]
        current_data = self.data
        for index, element in enumerate(elements):
            current_data = current_data.get(f'/{element}')
            if current_data is None:
                current_xpath = '/' + '/'.join(elements[:index + 1])
                raise RuntimeError(
                    f'Cannot find visual attributes for {current_xpath!r} ' +
                    f'while searching for {xpath!r} in {self.path!r}.')
        return current_data

def get_tag_name(xpath_element: str):
    """Extracts HTML tag name from `xpath_element` (strips the indexer)."""

    return re.match(XPATH_ELEMENT_REGEX, xpath_element).group(1)
