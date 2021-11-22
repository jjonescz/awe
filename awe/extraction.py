from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch_geometric import data as gdata

from awe import awe_graph, features

if TYPE_CHECKING:
    from awe.data import dataset


@dataclass
class IndirectData:
    """
    Wrapper around list to avoid default collate behavior of PyTorch Geometric.
    Used for `IndirectFeature`s that will be processed manually.
    """
    data: list

def collate(items: list[IndirectData]):
    return [x for item in items for x in item.data]

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
        return IndirectData([
            feature.compute(node, self.ctx) for node in self.ctx.nodes
        ])

    def get_node_label(self, node: awe_graph.HtmlNode):
        # Only the first label for now.
        label = None if len(node.labels) == 0 else node.labels[0]
        return self.ds.label_map[label]

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

        # Edges: parent-child relations.
        child_edges = [
            [node.dataset_index, child.dataset_index]
            for node in self.ctx.nodes for child in node.children
            # Ignore removed children.
            if self.ds.parent.node_predicate.include_node(child)
        ]
        parent_edges = [
            [node.dataset_index, node.parent.dataset_index]
            for node in self.ctx.nodes
            if (
                node.parent is not None and
                # Ignore removed parents.
                self.ds.parent.node_predicate.include_node(node.parent)
            )
        ]
        edge_index = torch.LongTensor(
            child_edges + parent_edges).t().contiguous()

        # Mask for "classifiable" nodes, i.e., leafs (text fragments).
        target = torch.BoolTensor([node.is_text for node in self.ctx.nodes])

        return gdata.Data(
            x=x,
            y=y,
            edge_index=edge_index,
            target=target,
            **indirect
        )
