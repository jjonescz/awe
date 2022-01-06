import itertools
from typing import Iterable

import torch
from torch_geometric import data, loader
from tqdm.auto import tqdm

from awe import awe_graph, awe_model, features
from awe.data import dataset


class Predictor:
    """Can do example predictions on a `Dataset`."""

    cached_batches: dict[int, data.Batch]
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
        self.cached_batches = {}
        self.cached_page_contexts = {}

    @property
    def items(self):
        return self.ds[self.name]

    def get_example_inputs(self, index: int):
        batch = self.cached_batches.get(index)
        if batch is None:
            batch = data.Batch.from_data_list([self.ds[self.name][index]])
            self.cached_batches[index] = batch
        return awe_model.ModelInputs(batch)

    def get_example(self, index: int):
        """Gets batch and its corresponding nodes."""
        page = self.items.pages[index]
        ctx = self.cached_page_contexts.get(page.identifier)
        if ctx is None:
            ctx = self.ds.prepare_page_context(page)
            self.cached_page_contexts[page.identifier] = ctx
        return self.get_example_inputs(index), ctx.nodes

    def evaluate_example(self, index: int, label: str):
        inputs = self.get_example_inputs(index)
        return self.model.compute_swde_metrics(
            inputs, self.ds.first_dataset.label_map[label])

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
            for label in tqdm(self.ds.first_dataset.label_map, desc='eval')
            if label is not None
        }

    def predict_example(self, index: int, label: str):
        """
        Gets predicted nodes for an example (and whether they were a match
        (i.e., a `True` positive) or not (`False`)).
        """
        inputs, nodes = self.get_example(index)

        predicted_nodes: list[tuple[bool, awe_graph.HtmlNode]] = []
        def handle(name: str, mask, idx=None):
            # Handle only positives (tp or fp).
            if name[1] == 'p':
                masked = itertools.compress(nodes, mask)
                node = next(itertools.islice(masked, idx, None))
                match = name[0] == 't' # is it true positive?
                predicted_nodes.append([match, node])
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
        return [n.text_content for n in self.ground_example(index, label)]

    def get_example_text(self, index: int, label: str):
        """Gets predicted and gold nodes for an example."""
        predicted = self.predict_example(index, label)
        if len(predicted) == 1:
            predicted = predicted[0]
            tp = predicted[0] # is it true positive?
            predicted = (tp, predicted[1].text_content)
            if tp:
                return predicted
        else:
            predicted = [(t[0], t[1].text_content) for t in predicted]
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
            for label in tqdm(self.ds.first_dataset.label_map, desc='texts')
            if label is not None
        }
