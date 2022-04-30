"""Model prediction decoding."""

from typing import TYPE_CHECKING

import pandas as pd

import awe.data.graph.dom
import awe.data.graph.pred
import awe.data.parsing
import awe.data.set.pages
import awe.model.classifier
import awe.utils

if TYPE_CHECKING:
    import awe.training.trainer


# pylint: disable=no-self-use
class Decoder:
    """Decodes model's outputs to predictions."""

    def __init__(self, trainer: 'awe.training.trainer.Trainer'):
        self.trainer = trainer
        self.pred_set = awe.data.graph.pred.PredictionSet(self.trainer)

    def gather(self, preds: list[awe.model.classifier.Prediction]):
        """Accumulates statistics of model predictions."""

        # Gather all node predictions.
        for pred in preds:
            self.pred_set.add_batch(pred)

    def get_label_keys(self):
        """Set of attribute keys."""

        return self.trainer.label_map.label_to_id.keys()

    def decode_raw(self, preds: list[awe.model.classifier.Prediction]):
        """Extracts all predicted nodes from model outputs."""

        self.gather(preds)
        return [
            {
                label_key: get_pred_nodes(label_key, page_pred)
                for label_key in self.get_label_keys()
            }
            for page_pred in self.pred_set.preds.values()
        ]

    def decode(self, preds: list[awe.model.classifier.Prediction]):
        """
        Like `decode_raw`, but displays the result in Pandas `DataFrame` for
        human inspection.
        """

        self.gather(preds)
        return pd.DataFrame(
            self.decode_page(page, page_pred)
            for page, page_pred in self.pred_set.preds.items()
        )

    def decode_page(self,
        page: awe.data.set.pages.Page,
        page_pred: awe.data.graph.pred.PredictedPage
    ):
        """Constructs `DataFrame` row for one page."""

        d = {
            'url': page.url
        }

        for label_key in self.get_label_keys():
            labeled_groups = page.dom.labeled_nodes.get(label_key)
            if labeled_groups is not None:
                d[f'gold_{label_key}'] = [
                    [self.decode_node(n) for n in group]
                    for group in labeled_groups
                ]
            pred_nodes = get_pred_nodes(label_key, page_pred)
            d[f'pred_{label_key}'] = [self.decode_node_pred(n) for n in pred_nodes]

        return d

    def decode_node(self, node: awe.data.graph.dom.Node):
        """Represents one `node` for displaying."""

        if node.is_text:
            return repr(awe.data.parsing.normalize_node_text(node.text))
        return f'<{node.html_tag}>'

    def decode_node_pred(self, pred: awe.data.graph.pred.NodePrediction):
        """
        Like `decode_node`, but includes also model's prediction confidence.
        """

        node = pred.node.find_node()
        return f'{self.decode_node(node)}({pred.confidence:.2f})'

def get_pred_nodes(
    label_key: str,
    page_pred: awe.data.graph.pred.PredictedPage
):
    """Orders predictions so the most confident are first."""

    return sorted(
        page_pred.preds.get(label_key, ()),
        key=lambda p: p.confidence,
        reverse=True,
    )
