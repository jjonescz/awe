"""Shared training context."""

import warnings

import awe.data.graph.dom

class LabelMap:
    """
    Map from attribute keys (a.k.a. labels) to IDs (a.k.a. classification
    classes).

    Note that ID 0 is reserved for `None` label.
    Hence, for N attribute keys, IDs are in range [0, N].
    """

    label_to_id: dict[str, int]
    id_to_label: dict[int, str]

    def __init__(self):
        self.label_to_id = {}
        self.id_to_label = {}

    def map_label_to_id(self, label: str):
        label_id = self.label_to_id.get(label)
        if label_id is None:
            label_id = len(self.label_to_id) + 1
            self.label_to_id[label] = label_id
            self.id_to_label[label_id] = label
        return label_id

    def get_label_id(self, node: awe.data.graph.dom.Node):
        if len(node.label_keys) == 0:
            return 0
        if len(node.label_keys) > 1:
            warnings.warn(
                f'More then one label key for {node.get_xpath()!r} ' +
                f'{node.label_keys!r} ({node.dom.page.html_path!r}).')
        return self.label_to_id[node.label_keys[0][0]]
