import itertools
from typing import Iterable

import torch
from torch_geometric import data, loader

from awe import awe_graph, awe_model
from awe.data import dataset


def get_texts(nodes: list[awe_graph.HtmlNode]):
    return list(map(lambda n: n.text_content, nodes))

class Predictor:
    """Can do example predictions on a `Dataset`."""

    def __init__(self,
        ds: dataset.DatasetCollection,
        name: str,
        model: awe_model.AweModel
    ):
        self.ds = ds
        self.name = name
        self.dataloader = loader.DataLoader(ds[name])
        self.model = model

    @property
    def items(self):
        return self.ds[self.name]

    def get_example(self, index: int):
        """Gets batch and its corresponding nodes."""
        batch: data.Batch = next(itertools.islice(self.dataloader, index, None))
        page = self.items.pages[index]
        ctx = self.ds.create_context(page)
        return batch, ctx.nodes

    def evaluate_example(self, index: int, label: str):
        batch, _ = self.get_example(index)
        return self.model.compute_swde_metrics(
            batch, self.ds.first_dataset.label_map[label])

    def evaluate_examples(self, indices: Iterable[int], label: str):
        total = torch.FloatTensor([0, 0, 0])
        count = 0
        for idx in indices:
            total += self.evaluate_example(idx, label).to_vector()
            count += 1
        return awe_model.SwdeMetrics.from_vector(total / count)

    def evaluate(self, indices: Iterable[int]):
        return {
            label: self.evaluate_examples(indices, label)
            for label in self.ds.first_dataset.label_map
            if label is not None
        }

    def predict_example(self, index: int, label: str):
        """Gets predicted nodes for an example."""
        batch, nodes = self.get_example(index)

        predicted_nodes: list[awe_graph.HtmlNode] = []
        def handle(name: str, mask, idx=None):
            # Handle only positives (tp or fp).
            if name[1] == 'p':
                masked = itertools.compress(nodes, mask)
                node = next(itertools.islice(masked, idx, None))
                predicted_nodes.append(node)
        self.model.predict_swde(
            batch, self.ds.first_dataset.label_map[label], handle)

        return predicted_nodes

    def ground_example(self, index: int, label: str):
        """Gets gold nodes for an example."""
        page = self.items.pages[index]
        nodes = page.labels.get_nodes(label)
        return nodes

    def ground_texts(self, index: int, label: str):
        """Gets gold texts for an example."""
        page = self.items.pages[index]
        texts = page.get_groundtruth_texts(label)
        if texts is not None:
            return texts
        return get_texts(self.ground_example(index, label))

    def get_example_text(self, index: int, label: str):
        """Gets predicted and gold nodes for an example."""
        predicted = get_texts(self.predict_example(index, label))
        if len(predicted) == 1:
            predicted = predicted[0]
        ground = self.ground_texts(index, label)
        if len(ground) == 1:
            ground = ground[0]
        return predicted, ground

    def get_example_texts(self, indices: Iterable[int]):
        return {
            label: [
                self.get_example_text(i, label)
                for i in indices
            ]
            for label in self.ds.first_dataset.label_map if label is not None
        }
