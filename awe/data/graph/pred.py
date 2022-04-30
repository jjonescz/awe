"""Helpers for DOM node predictions."""

import collections
import dataclasses
from typing import TYPE_CHECKING

import torch

import awe.data.set.pages
import awe.model.classifier

if TYPE_CHECKING:
    import awe.data.graph.dom
    import awe.training.trainer


@dataclasses.dataclass
class NodePrediction:
    """Represents one node prediction of the model."""

    node: 'awe.data.graph.dom.NodeIdentity'
    """The predicted node."""

    confidence: float
    """Model's un-normalized confidence (logit)."""

    probability: float
    """Normalized `confidence`."""

class PredictedPage:
    """Stores predicted nodes for each label in a page."""

    preds: dict[str, list[NodePrediction]]
    """Predicted nodes for each label key."""

    num_predicted = 0
    """Total number of predicted nodes."""

    def __init__(self):
        self.preds = collections.defaultdict(list)

    def add(self, label_key: str, pred: NodePrediction):
        """Marks a node as predicted with the given `label_key`."""

        self.preds[label_key].append(pred)

    def increment(self):
        """Marks a new node as predicted in this page."""

        self.num_predicted += 1

    def clear(self):
        """Clears all state."""

        for l in self.preds.values():
            l.clear()
        self.num_predicted = 0

class PredictionSet:
    """Stores predicted nodes for each page."""

    preds: dict[awe.data.set.pages.Page, PredictedPage]
    """Predicted nodes for each page."""

    def __init__(self, trainer: 'awe.training.trainer.Trainer'):
        self.trainer = trainer
        self.preds = {}

    def add(self, label_key: str, pred: NodePrediction):
        """Marks a node as predicted with the given `label_key`."""

        self.get_or_add(pred.node.page).add(label_key, pred)

    def increment(self, node: 'awe.data.graph.dom.Node'):
        """Marks the `node` as predicted."""

        self.get_or_add(node.dom.page).increment()

    def get_or_add(self, page: awe.data.set.pages.Page):
        """Obtains `PredictedPage` for the given `page`."""

        pred_page = self.preds.get(page)
        if pred_page is None:
            pred_page = PredictedPage()
            self.preds[page] = pred_page
        return pred_page

    def clear(self):
        """Clears all state."""

        for pred_page in self.preds.values():
            pred_page.clear()

    def add_batch(self, pred: awe.model.classifier.Prediction):
        """Populates state given a model's prediction (`pred`)."""

        pred_labels = pred.outputs.get_pred_labels()
        probabilities = pred.outputs.get_probabilities()
        for idx in torch.nonzero(pred_labels):
            label_id = pred_labels[idx]
            label_key = self.trainer.label_map.id_to_label[label_id.item()]
            pred_node: awe.data.graph.dom.Node = pred.batch[idx.item()]
            node_pred = NodePrediction(
                node=pred_node.get_identity(),
                confidence=pred.outputs.logits[idx, label_id].item(),
                probability=probabilities[idx, label_id].item()
            )
            self.add(label_key=label_key, pred=node_pred)
        for node in pred.batch:
            self.increment(node)
