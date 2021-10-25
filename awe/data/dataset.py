from typing import Optional

import joblib
import torch
from torch_geometric import data as gdata
from torch_geometric import loader
from tqdm.auto import tqdm

from awe import awe_graph, features


class Dataset:
    label_map: Optional[dict[Optional[str], int]] = None
    data: dict[str, list[gdata.Data]] = {}
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
            return gdata.Data(x=x, y=y)

        return list(joblib.Parallel(n_jobs=2)(
            map(joblib.delayed(prepare_page), tqdm(pages, desc='pages'))))

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
        return sum(len(f.labels) for f in self.features)

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
                    curr_nodes = list(page.nodes)
                node = curr_nodes[node_idx + node_offset]

                yield node, batch.x[node_idx], batch.y[node_idx]
            page_idx += batch.num_graphs
