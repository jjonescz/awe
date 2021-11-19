import collections
import functools
import os
from typing import Callable, Optional

import torch
from torch_geometric import data as gdata
from torch_geometric import loader as gloader

from awe import awe_graph
from awe import features as f
from awe import filtering, utils


# Implements PyG dataset API, see
# https://pytorch-geometric.readthedocs.io/en/2.0.1/notes/create_dataset.html.
class Dataset:
    shuffle = False
    label_map: Optional[dict[Optional[str], int]] = None
    loader: Optional[gloader.DataLoader] = None
    in_memory_data: dict[int, gdata.Data] = {}

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

    def __getitem__(self, idx: int) -> gdata.Data:
        page = self.pages[idx]
        if page.data_point_path is None:
            return self.in_memory_data[idx]
        else:
            return torch.load(page.data_point_path)

    def __len__(self):
        return len(self.pages)

    def _prepare_page_context(self, page: awe_graph.HtmlPage):
        ctx = self.parent.create_page_context(page)

        # Assign indices to nodes (different from `HtmlNode.index` as that
        # one is from before filtering). This is needed to compute edges.
        for index, node in enumerate(ctx.nodes):
            node.dataset_index = index

        return ctx

    def prepare_page_features(self, idx: int):
        """Prepares features for page at `idx`."""
        page = self.pages[idx]
        ctx = self._prepare_page_context(page)

        for feature in self.parent.features:
            for node in ctx.nodes:
                feature.prepare(node, ctx.root)

    def compute_page_features(self, idx: int):
        """Computes features for page at `idx` and persists them on disk."""
        page = self.pages[idx]
        ctx = self._prepare_page_context(page)

        def compute_node_features(node: awe_graph.HtmlNode):
            return torch.hstack([
                feature.compute(node, ctx)
                for feature in self.parent.features
            ])

        def get_node_label(node: awe_graph.HtmlNode):
            # Only the first label for now.
            label = None if len(node.labels) == 0 else node.labels[0]
            return self.label_map[label]

        x = torch.vstack(list(map(compute_node_features, ctx.nodes)))
        y = torch.tensor(list(map(get_node_label, ctx.nodes)))

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

        # Mask for "classifiable" nodes, i.e., leafs (text fragments).
        target = torch.BoolTensor([node.is_text for node in ctx.nodes])

        data = gdata.Data(x=x, y=y, edge_index=edge_index, target=target)
        if page.data_point_path is None:
            self.in_memory_data[idx] = data
        else:
            torch.save(data, page.data_point_path)

    def will_prepare_page(self, idx: int, skip_existing=True):
        """Determines whether this page needs features to be computed."""
        if (
            not skip_existing or
            not os.path.exists(self.pages[idx].data_point_path)
        ):
            return True
        return False

    def will_prepare_any(self, skip_existing=True):
        """Determines whether any page features will be computed."""
        for i in range(len(self)):
            if self.will_prepare_page(i, skip_existing):
                return True
        return False

    def _process_features(self,
        processor: Callable[[int]],
        parallelize: Optional[int] = None,
        skip_existing: bool = True
    ):
        pages_to_process = list(filter(
            functools.partial(self.will_prepare_page,
                skip_existing=skip_existing),
            range(len(self))
        ))
        if len(pages_to_process) != 0:
            utils.parallelize(
                parallelize,
                processor,
                pages_to_process,
                self.name
            )
        return len(pages_to_process)

    def prepare_features(self,
        parallelize: Optional[int] = None,
        skip_existing: bool = True
    ):
        """Prepares features for all pages where necessary."""
        return self._process_features(self.prepare_page_features,
            parallelize=parallelize,
            skip_existing=skip_existing
        )

    def compute_features(self,
        parallelize: Optional[int] = None,
        skip_existing: bool = True
    ):
        """Computes features for all pages where necessary."""
        return self._process_features(self.compute_page_features,
            parallelize=parallelize,
            skip_existing=skip_existing
        )

    def delete_saved(self):
        """Deletes saved computed features (backup file `.bak` is created)."""
        counter = 0
        for page in self.pages:
            pt_path = page.data_point_path
            if pt_path is not None and os.path.exists(pt_path):
                os.rename(pt_path, f'{pt_path}.bak')
                counter += 1
        return counter

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
                    curr_ctx = self.parent.create_page_context(page)
                node = curr_ctx.nodes[node_idx + node_offset]

                yield node, batch.x[node_idx], batch.y[node_idx]
            page_idx += batch.num_graphs

    def count_labels(self):
        counts = collections.defaultdict(int)
        for page in self.pages:
            for label in self.parent.first_dataset.label_map.keys():
                if label is not None:
                    counts[label] += page.count_label(label)
        return counts

class DatasetCollection:
    features: list[f.Feature] = []
    node_predicate: filtering.NodePredicate = filtering.DefaultNodePredicate()
    first_dataset: Optional[Dataset] = None
    datasets: dict[str, Dataset] = {}

    def __init__(self):
        self.root = f.RootContext()

    def __getitem__(self, name: str):
        return self.datasets[name]

    def create(self,
        name: str, pages: list[awe_graph.HtmlPage], shuffle = False
    ):
        ds = Dataset(name, self, pages, self.first_dataset)
        ds.shuffle = shuffle
        self.datasets[name] = ds
        if self.first_dataset is None:
            self.first_dataset = ds
        return ds

    def create_page_context(self, page: awe_graph.HtmlPage):
        ctx = f.PageContext(self.root, page, self.node_predicate)
        page.prepare(ctx)
        return ctx

    def will_prepare_any(self, skip_existing=True):
        """Determines whether any page features will be computed."""
        for ds in self.datasets.values():
            if ds.will_prepare_any(skip_existing):
                return True
        return False

    def initialize(self, skip_existing=True):
        """Initializes features if necessary."""
        if self.will_prepare_any(skip_existing):
            for feature in self.features:
                feature.initialize()
            return True
        return False

    @property
    def feature_dim(self):
        """Feature vector total length."""
        return sum(f.dimension for f in self.features)

    @property
    def feature_labels(self):
        """Description of each feature vector column."""
        return [label for f in self.features for label in f.labels]

    def _initialize(self,
        parallelize: Optional[int] = None,
        skip_existing: bool = True
    ):
        if parallelize is None:
            # TODO: Initialization won't have effect on other cores, hence it's
            # skipped if parallelization is enabled.
            self.initialize(skip_existing=skip_existing)

    def prepare_features(self,
        parallelize: Optional[int] = None,
        skip_existing: bool = True
    ):
        self._initialize(parallelize=parallelize, skip_existing=skip_existing)
        counter = 0
        for ds in self.datasets.values():
            counter += ds.prepare_features(
                parallelize=parallelize,
                skip_existing=skip_existing
            )
        return counter

    def compute_features(self,
        parallelize: Optional[int] = None,
        skip_existing: bool = True
    ):
        self._initialize(parallelize=parallelize, skip_existing=skip_existing)
        counter = 0
        for ds in self.datasets.values():
            counter += ds.compute_features(
                parallelize=parallelize,
                skip_existing=skip_existing
            )
        return counter

    def delete_saved_features(self):
        counter = 0
        for ds in self.datasets.values():
            counter += ds.delete_saved()
        return counter

    def count_labels(self):
        return {
            name: ds.count_labels()
            for name, ds in self.datasets.items()
        }

    def get_lengths(self):
        return {
            name: len(ds)
            for name, ds in self.datasets.items()
        }

    def create_dataloaders(self, *, batch_size: int = 1, num_workers: int = 0):
        for ds in self.datasets.values():
            ds.loader = gloader.DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=ds.shuffle,
                num_workers=num_workers
            )

    def _iterate_pages_without_visual_features(self):
        for ds in self.datasets.values():
            for page in ds.pages:
                if not page.has_dom_data:
                    yield page

    def summarize_pages_without_visual_features(self):
        return utils.summarize_pages(
            self._iterate_pages_without_visual_features())
