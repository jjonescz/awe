import itertools
from typing import Iterable

import torch
from torch_geometric import data, loader
from tqdm.auto import tqdm

from awe import awe_graph, awe_model, features
from awe.data import dataset


def get_texts(nodes: list[awe_graph.HtmlNode]):
    return list(map(lambda n: n.text_content, nodes))

class Predictor:
    """Can do example predictions on a `Dataset`."""

    cached_page_contexts: dict[str, features.PageContext]

    def __init__(self,
        ds: dataset.DatasetCollection,
        name: str,
        model: awe_model.AweModel
    ):
        self.ds = ds
        self.name = name
        self.dataloader = loader.DataLoader(ds[name])
        self.model = model
        self.cached_page_contexts = {}

    @property
    def items(self):
        return self.ds[self.name]

    def get_example(self, index: int):
        """Gets batch and its corresponding nodes."""
        batch: data.Batch = next(itertools.islice(self.dataloader, index, None))
        page = self.items.pages[index]
        ctx = self.cached_page_contexts.get(page.identifier)
        if ctx is None:
            ctx = self.ds.prepare_page_context(page)
            self.cached_page_contexts[page.identifier] = ctx
        inputs = awe_model.ModelInputs(batch)
        return inputs, ctx.nodes

    def evaluate_example(self, index: int, label: str):
        inputs, _ = self.get_example(index)
        return self.model.compute_swde_metrics(
            inputs, self.ds.first_dataset.label_map[label])

    def evaluate_examples(self, indices: Iterable[int], label: str):
        total = torch.FloatTensor([0, 0, 0])
        count = 0
        for idx in tqdm(indices, leave=False, desc='pages'):
            total += self.evaluate_example(idx, label).to_vector()
            count += 1
        return awe_model.SwdeMetrics.from_vector(total / count)

    def evaluate(self, indices: Iterable[int]):
        return {
            label: self.evaluate_examples(indices, label)
            for label in tqdm(self.ds.first_dataset.label_map, desc='labels')
            if label is not None
        }

    def predict_example(self, index: int, label: str):
        """Gets predicted nodes for an example."""
        inputs, nodes = self.get_example(index)

        predicted_nodes: list[awe_graph.HtmlNode] = []
        def handle(name: str, mask, idx=None):
            # Handle only positives (tp or fp).
            if name[1] == 'p':
                masked = itertools.compress(nodes, mask)
                node = next(itertools.islice(masked, idx, None))
                predicted_nodes.append(node)
        self.model.predict_swde(
            inputs, self.ds.first_dataset.label_map[label], handle)

        return predicted_nodes

    def ground_example(self, index: int, label: str):
        """Gets gold nodes for an example."""
        page = self.items.pages[index]
        page_nodes = page.get_tree().descendants
        ground_nodes = page.labels.get_nodes(label, page_nodes)
        return ground_nodes

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
        if predicted == ground:
            return predicted
        return predicted, ground

    def get_example_texts(self, indices: Iterable[int]):
        return {
            label: [
                self.get_example_text(i, label)
                for i in tqdm(indices, leave=False, desc='pages')
            ]
            for label in tqdm(self.ds.first_dataset.label_map, desc='labels')
            if label is not None
        }
