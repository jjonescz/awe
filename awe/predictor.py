import itertools
from typing import Iterable

import torch
from torch_geometric import loader

from awe import awe_graph, awe_model
from awe.data import dataset


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

    def get_example(self, index: int):
        example_batch = next(itertools.islice(self.dataloader, index, None))
        example_page = self.ds[self.name].pages[index]
        example_ctx = self.ds.create_context(example_page)
        return example_batch, example_ctx.nodes

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

    def predict_example(self, index: int, label: str) -> list[awe_graph.HtmlNode]:
        batch, nodes = self.get_example(index)

        predicted_nodes = []
        def handle(name: str, mask, idx=None):
            if name[1] == 'p':
                masked = itertools.compress(nodes, mask)
                node = next(itertools.islice(masked, idx, None))
                predicted_nodes.append(node)
        self.model.predict_swde(
            batch, self.ds.first_dataset.label_map[label], handle)

        return predicted_nodes

    def get_example_texts(self, indices: Iterable[int]):
        return {
            label: [
                [node.text_content for node in self.predict_example(i, label)]
                for i in indices
            ]
            for label in self.ds.first_dataset.label_map if label is not None
        }
