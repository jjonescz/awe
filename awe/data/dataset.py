import collections
from typing import Optional

import torch
from torch_geometric.data import Data

from awe import awe_graph, features


def _new_label_id_counter():
    counter = 0
    def new_label_id():
        nonlocal counter
        counter += 1
        return counter
    return new_label_id

def _create_label_map():
    label_map = collections.defaultdict(_new_label_id_counter())
    label_map[None] = 0
    return label_map

class Dataset:
    label_map: Optional[dict[Optional[str], int]]
    data: dict[str, list[Data]]
    pages: dict[str, list[awe_graph.HtmlPage]]

    def __init__(self):
        self.label_map = None
        self.data = {}
        self.pages = {}

    def _prepare_data(self, pages: list[awe_graph.HtmlPage]):
        def get_node_features(node: awe_graph.HtmlNode):
            categories = node.get_feature(features.CharCategories)
            return [
                categories.dollars,
                categories.letters,
                categories.digits,
                node.get_feature(features.Depth).relative
            ]

        def get_node_label(node: awe_graph.HtmlNode):
            # Only the first label for now.
            label = None if len(node.labels) == 0 else node.labels[0]
            return self.label_map[label]

        def prepare_page(page: awe_graph.HtmlPage):
            ctx = features.FeatureContext(page)
            ctx.add_all([
                features.CharCategories,
                features.Depth
            ])
            x = torch.tensor(list(map(get_node_features, ctx.nodes)))
            y = torch.tensor(list(map(get_node_label, ctx.nodes)))
            return Data(x=x, y=y)

        return list(map(prepare_page, pages))

    def add(self, name: str, pages: list[awe_graph.HtmlPage]):
        if self.label_map is None:
            self.label_map = _create_label_map()
        else:
            # Freeze label map.
            self.label_map.default_factory = None
        self.pages[name] = pages
        self.data[name] = self._prepare_data(pages)

    @property
    def feature_count(self):
        return self.data['train'][0].x.shape[1]
