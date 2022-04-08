import awe.data.graph.dom

class LabelMap:
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
        return self.label_to_id[node.label_keys[0][0]]
