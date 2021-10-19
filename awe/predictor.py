import itertools
from typing import Iterable

from torch_geometric import loader

from awe import awe_model, awe_graph
from awe.data import dataset


class Predictor:
    """Can do example predictions on a `Dataset`."""

    def __init__(self, ds: dataset.Dataset, name: str, model: awe_model.AweModel):
        self.ds = ds
        self.name = name
        self.dataloader = loader.DataLoader(ds.data[name])
        self.model = model

    def get_example(self, index: int):
        example_batch = next(itertools.islice(self.dataloader, index, None))
        example_page = self.ds.pages[self.name][index]
        example_nodes = list(example_page.nodes)
        return example_batch, example_nodes

    def evaluate_example(self, index: int, label: str):
        batch, _ = self.get_example(index)
        return self.model.compute_swde_metrics(batch, self.ds.label_map[label])

    def evaluate(self, indices: Iterable[int]):
        return [{
            label: self.evaluate_example(i, label)
            for label in self.ds.label_map
            if label is not None
        } for i in indices]

    def predict_example(self, index: int, label: str) -> list[awe_graph.HtmlNode]:
        batch, nodes = self.get_example(index)

        predicted_nodes = []
        def handle(name: str, mask, idx=None):
            if name[1] == 'p':
                masked = itertools.compress(nodes, mask)
                node = next(itertools.islice(masked, idx, None))
                predicted_nodes.append(node)
        self.model.predict_swde(batch, self.ds.label_map[label], handle)

        return predicted_nodes

    def get_example_texts(self, indices: Iterable[int]):
        return {
            label: [
                [node.text_content for node in self.predict_example(i, label)]
                for i in indices
            ]
            for label in self.ds.label_map if label is not None
        }
