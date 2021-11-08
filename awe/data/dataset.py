from typing import Callable, Optional

import torch
from lxml import etree
from torch_geometric import data as gdata
from torch_geometric import loader

from awe import awe_graph, features, utils


class Dataset:
    @staticmethod
    def default_node_predicate(node: awe_graph.HtmlNode):
        return node.is_text or not (
            node.element.tag is etree.Comment or
            node.element.tag in ['script', 'style', 'noscript']
        )

    features: list['features.Feature'] = []
    label_map: Optional[dict[Optional[str], int]] = None
    data: dict[str, list[gdata.Data]] = {}
    pages: dict[str, list[awe_graph.HtmlPage]] = {}
    loaders: dict[str, loader.DataLoader] = {}
    parallelize = 2
    node_predicate: Callable[[awe_graph.HtmlNode], bool] = default_node_predicate

    def get_context(self, page: awe_graph.HtmlPage):
        ctx = features.FeatureContext(page, self.node_predicate)
        return ctx

    def _prepare_data(self, pages: list[awe_graph.HtmlPage]):
        def prepare_page(page: awe_graph.HtmlPage):
            ctx = self.get_context(page)

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

            # Assign indices to nodes (different from `HtmlNode.index` as that
            # one is from before filtering). This is needed to compute edges.
            for index, node in enumerate(ctx.nodes):
                node.dataset_index = index

            # Edges: parent-child relations.
            child_edges = [
                [node.dataset_index, child.dataset_index]
                for node in ctx.nodes for child in node.children
            ]
            parent_edges = [
                [node.dataset_index, node.parent.dataset_index]
                for node in ctx.nodes
                if (
                    node.parent is not None and
                    # Ignore removed parents.
                    self.node_predicate(node.parent)
                )
            ]
            edge_index = torch.LongTensor(
                child_edges + parent_edges).t().contiguous()

            return gdata.Data(x=x, y=y, edge_index=edge_index)

        return utils.parallelize(self.parallelize, prepare_page, pages, 'pages')

    def add(self, name: str, pages: list[awe_graph.HtmlPage]):
        if self.label_map is None:
            # Create label map.
            self.label_map = { None: 0 }
            label_counter = 1
            for page in pages:
                for field in page.fields:
                    if field not in self.label_map:
                        self.label_map[field] = label_counter
                        label_counter += 1
        else:
            # Check label map.
            for page in pages:
                for field in page.fields:
                    if field not in self.label_map:
                        raise ValueError(f'Field {field} from page {page} ' +
                            'not found in the label map.')
        self.pages[name] = pages
        self.data[name] = self._prepare_data(pages)

    @property
    def feature_dim(self):
        """Feature vector total length."""
        return sum(f.dimension for f in self.features)

    @property
    def feature_labels(self):
        """Description of each feature vector column."""
        return [label for f in self.features for label in f.labels]

    def iterate_data(self, name: str):
        """Iterates `HtmlNode`s along with their feature vectors and labels."""
        page_idx = 0
        for batch in self.loaders[name]:
            curr_page = None
            curr_nodes = None
            node_offset = 0
            prev_page = 0
            for node_idx in range(batch.num_nodes):
                page_offset = batch.batch[node_idx]

                if prev_page != page_offset:
                    assert prev_page == page_offset - 1
                    prev_page = page_offset
                    node_offset = -node_idx

                page = self.pages[name][page_idx + page_offset]
                if curr_page != page:
                    curr_page = page
                    if self.node_predicate is None:
                        curr_nodes = list(page.nodes)
                    else:
                        curr_nodes = list(
                            filter(self.node_predicate, page.nodes))
                node = curr_nodes[node_idx + node_offset]

                yield node, batch.x[node_idx], batch.y[node_idx]
            page_idx += batch.num_graphs
