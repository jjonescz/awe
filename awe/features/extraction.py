from typing import TYPE_CHECKING

import torch
from torch_geometric import data as gdata

from awe import awe_graph, features, graph_utils

if TYPE_CHECKING:
    from awe.data import dataset


class PageFeatureExtractor:
    """Can extract features for one `HtmlPage`."""

    def __init__(self,
        ds: 'dataset.Dataset',
        ctx: features.PageContext
    ):
        self.ds = ds
        self.ctx = ctx

    def compute_direct_features(self, node: awe_graph.HtmlNode):
        return torch.hstack([
            feature.compute(node, self.ctx)
            for feature in self.ds.parent.features
            if isinstance(feature, features.DirectFeature)
        ])

    def compute_indirect_features(self, feature: features.IndirectFeature):
        vectors = [feature.compute(node, self.ctx) for node in self.ctx.nodes]
        return torch.stack(vectors)

    def get_node_label(self, node: awe_graph.HtmlNode):
        # Only the first label for now.
        label = None if len(node.labels) == 0 else node.labels[0]
        return self.ds.label_map[label]

    def describe_feature(self,
        node: awe_graph.HtmlNode,
        feature: features.Feature
    ):
        vector = feature.compute(node, self.ctx)
        if isinstance(feature, features.DirectFeature):
            return (feature.__class__.__name__, dict(zip(feature.labels, vector)))
        if isinstance(feature, features.IndirectFeature):
            return (feature.label, vector)
        return (feature.__class__.__name__, None)

    def describe(self):
        """Extracts features into a `dict` that can be inspected."""

        self.ds.parent.initialize_features()

        return {
            node.xpath: dict(
                self.describe_feature(node, feature)
                for feature in self.ds.parent.features
            )
            for node in self.ctx.nodes
        }

    def extract(self):
        """Extracts features to `Data` instance that can be serialized."""

        self.ds.parent.initialize_features()

        # Compute direct node features and labels.
        x = torch.vstack(list(map(self.compute_direct_features, self.ctx.nodes)))
        y = torch.tensor(list(map(self.get_node_label, self.ctx.nodes)))

        # Compute indirect node features.
        indirect = {
            f.label: self.compute_indirect_features(f)
            for f in self.ds.parent.features
            if isinstance(f, features.IndirectFeature)
        }

        # Compute edges.
        pg = graph_utils.PageGraph(self.ctx)
        pg.link_children_or_parents()
        edge_index = pg.get_edge_index()

        # Mask for "classifiable" nodes, i.e., leafs (text fragments).
        target = torch.BoolTensor([node.is_text for node in self.ctx.nodes])

        return gdata.Data(
            x=x,
            y=y,
            edge_index=edge_index,
            target=target,
            **indirect
        )
