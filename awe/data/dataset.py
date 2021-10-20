import collections
from typing import Optional

import torch
from torch_geometric import loader
from torch_geometric.data import Data
from tqdm.auto import tqdm

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
    label_map: Optional[dict[Optional[str], int]] = None
    data: dict[str, list[Data]] = {}
    pages: dict[str, list[awe_graph.HtmlPage]] = {}
    loaders: dict[str, loader.DataLoader] = {}

    def __init__(self, fs: list[features.Feature]):
        self.features = fs

    def _prepare_data(self, pages: list[awe_graph.HtmlPage]):
        def prepare_page(page: awe_graph.HtmlPage):
            ctx = features.FeatureContext(page)

            def get_node_features(node: awe_graph.HtmlNode):
                return torch.hstack([
                    feature.create(node, ctx)
                    for feature in self.features
                ])

            def get_node_label(node: awe_graph.HtmlNode):
                # Only the first label for now.
                label = None if len(node.labels) == 0 else node.labels[0]
                return self.label_map[label]

            x = torch.vstack(list(map(get_node_features, ctx.nodes)))
            y = torch.tensor(list(map(get_node_label, ctx.nodes)))
            return Data(x=x, y=y)

        return list(map(prepare_page, tqdm(pages, desc='pages')))

    def add(self, name: str, pages: list[awe_graph.HtmlPage]):
        if self.label_map is None:
            self.label_map = _create_label_map()
        else:
            # Freeze label map.
            self.label_map.default_factory = None
        self.pages[name] = pages
        self.data[name] = self._prepare_data(pages)

    @property
    def feature_dim(self):
        """Feature vector total length."""
        return sum(len(f.labels) for f in self.features)

    @property
    def feature_labels(self):
        """Description of each feature vector column."""
        return [label for f in self.features for label in f.labels]
