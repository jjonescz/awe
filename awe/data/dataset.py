import os
from typing import Optional

import torch
from torch_geometric import data as gdata
from torch_geometric import loader as gloader

from awe import awe_graph
from awe import features as f
from awe import filtering, utils


class Dataset:
    label_map: Optional[dict[Optional[str], int]] = None
    loader: Optional[gloader.DataLoader] = None
    parallelize: Optional[int] = None

    def __init__(self,
        name: str,
        parent: 'DatasetCollection',
        pages: list[awe_graph.HtmlPage],
        other: Optional['Dataset'] = None
    ):
        self.name = name
        self.parent = parent
        self.pages = pages
        if other is not None:
            self.label_map = other.label_map
        self._prepare_label_map()

    def __getitem__(self, idx: int):
        page = self.pages[idx]
        return torch.load(page.data_point_path)

    def prepare_page(self, page: awe_graph.HtmlPage):
        """Computes features for `page` and persists them on disk."""
        ctx = self.parent.create_context(page)

        def get_node_features(node: awe_graph.HtmlNode):
            return torch.hstack([
                feature.create(node, ctx)
                for feature in self.parent.features
            ])

        def get_node_label(node: awe_graph.HtmlNode):
            # Only the first label for now.
            label = None if len(node.labels) == 0 else node.labels[0]
            return self.label_map[label]

        x = torch.vstack(list(map(get_node_features, ctx.nodes)))
        y = torch.tensor(list(map(get_node_label, ctx.nodes)))

        # Assign indices to nodes (different from `HtmlNode.index` as that
        # one is from before filtering). This is needed to compute edges.
        for index, node in enumerate(ctx.nodes):
            node.dataset_index = index

        # Edges: parent-child relations.
        child_edges = [
            [node.dataset_index, child.dataset_index]
            for node in ctx.nodes for child in node.children
            # Ignore removed children.
            if self.parent.node_predicate.include_node(child)
        ]
        parent_edges = [
            [node.dataset_index, node.parent.dataset_index]
            for node in ctx.nodes
            if (
                node.parent is not None and
                # Ignore removed parents.
                self.parent.node_predicate.include_node(node.parent)
            )
        ]
        edge_index = torch.LongTensor(
            child_edges + parent_edges).t().contiguous()

        data = gdata.Data(x=x, y=y, edge_index=edge_index)
        torch.save(data, page.data_point_path)

    def prepare(self, skip_existing=False):
        def prepare_one(page: awe_graph.HtmlPage):
            if not skip_existing or not os.path.exists(page.data_point_path):
                self.prepare_page(page)
        utils.parallelize(self.parallelize, prepare_one, self.pages, 'pages')
        return len(self.pages)

    def _prepare_label_map(self):
        if self.label_map is None:
            # Create label map.
            self.label_map = { None: 0 }
            label_counter = 1
            for page in self.pages:
                for field in page.fields:
                    if field not in self.label_map:
                        self.label_map[field] = label_counter
                        label_counter += 1
        else:
            # Check label map.
            for page in self.pages:
                for field in page.fields:
                    if field not in self.label_map:
                        raise ValueError(f'Field {field} from page {page} ' +
                            'not found in the label map.')

    def iterate_data(self):
        """Iterates `HtmlNode`s along with their feature vectors and labels."""
        page_idx = 0
        for batch in self.loader or []:
            curr_page = None
            curr_ctx = None
            node_offset = 0
            prev_page = 0
            for node_idx in range(batch.num_nodes):
                page_offset = batch.batch[node_idx]

                if prev_page != page_offset:
                    assert prev_page == page_offset - 1
                    prev_page = page_offset
                    node_offset = -node_idx

                page = self.pages[page_idx + page_offset]
                if curr_page != page:
                    curr_page = page
                    curr_ctx = self.parent.create_context(page)
                node = curr_ctx.nodes[node_idx + node_offset]

                yield node, batch.x[node_idx], batch.y[node_idx]
            page_idx += batch.num_graphs

class DatasetCollection:
    features: list[f.Feature] = []
    node_predicate: filtering.NodePredicate = filtering.DefaultNodePredicate()
    first_dataset: Optional[Dataset] = None
    datasets: dict[str, Dataset] = {}

    def __getitem__(self, name: str):
        return self.datasets[name]

    def create(self, name: str, pages: list[awe_graph.HtmlPage]):
        ds = Dataset(name, self, pages, self.first_dataset)
        self.datasets[name] = ds
        if self.first_dataset is None:
            self.first_dataset = ds
        return ds

    def create_context(self, page: awe_graph.HtmlPage):
        ctx = f.FeatureContext(page, self.node_predicate)
        page.prepare(ctx)
        return ctx

    @property
    def feature_dim(self):
        """Feature vector total length."""
        return sum(f.dimension for f in self.features)

    @property
    def feature_labels(self):
        """Description of each feature vector column."""
        return [label for f in self.features for label in f.labels]
