import collections
import dataclasses
from typing import TYPE_CHECKING

import awe.data.set.pages

if TYPE_CHECKING:
    import awe.data.graph.dom


@dataclasses.dataclass
class NodePrediction:
    node: 'awe.data.graph.dom.Node'
    confidence: float

class PredictedPage:
    """Stores predicted nodes for each label in a page."""

    preds: dict[str, list[NodePrediction]]
    num_predicted = 0

    def __init__(self):
        self.preds = collections.defaultdict(list)

    def add(self, label_key: str, pred: NodePrediction):
        l = self.preds[label_key]
        l.append(pred)
        l.sort(key=lambda p: p.confidence, reverse=True)

    def increment(self):
        self.num_predicted += 1

    def clear(self):
        for l in self.preds.values():
            l.clear()
        self.num_predicted = 0

class PredictionSet:
    """Stores predicted nodes for each page."""

    preds: dict[awe.data.set.pages.Page, PredictedPage]

    def __init__(self):
        self.preds = {}

    def add(self, label_key: str, pred: NodePrediction):
        self.get_or_add(pred.node.dom.page).add(label_key, pred)

    def increment(self, node: 'awe.data.graph.dom.Node'):
        self.get_or_add(node.dom.page).increment()

    def get_or_add(self, page: awe.data.set.pages.Page):
        pred_page = self.preds.get(page)
        if pred_page is None:
            pred_page = PredictedPage()
            self.preds[page] = pred_page
        return pred_page

    def clear(self):
        for pred_page in self.preds.values():
            pred_page.clear()
