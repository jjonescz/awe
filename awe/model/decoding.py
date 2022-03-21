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
    """Can decode model's predictions for human inspection."""

    def __init__(self, trainer: 'awe.training.trainer.Trainer'):
        self.trainer = trainer
        self.pred_set = awe.data.graph.pred.PredictionSet(self.trainer)

    def decode(self, preds: list[awe.model.classifier.Prediction]):
        # Gather all node predictions.
        for pred in preds:
            self.pred_set.add_batch(pred)

        return pd.DataFrame(
            self.decode_one(page, page_pred)
            for page, page_pred in self.pred_set.preds.items()
        )

    def decode_one(self,
        page: awe.data.set.pages.Page,
        page_pred: awe.data.graph.pred.PredictedPage
    ):
        d = {
            # 'vertical': page.website.vertical.name,
            # 'website': page.website.name,
            # 'index': page.index,
            'url': page.url
        }

        for label_key in self.trainer.label_map.label_to_id.keys():
            labeled_nodes = page.dom.labeled_nodes.get(label_key)
            if labeled_nodes is not None:
                d[f'gold_{label_key}'] = [self.decode_node(n) for n in labeled_nodes]

            # Sort predictions by most confident.
            pred_nodes = sorted(
                page_pred.preds.get(label_key, ()),
                key=lambda p: p.confidence,
                reverse=True,
            )

            d[f'pred_{label_key}'] = [self.decode_node_pred(n) for n in pred_nodes]

        return d

    def decode_node(self, node: awe.data.graph.dom.Node):
        if node.is_text:
            return repr(awe.data.parsing.normalize_node_text(node.text))
        return f'<{node.html_tag}>'

    def decode_node_pred(self, pred: awe.data.graph.pred.NodePrediction):
        return f'{self.decode_node(pred.node)}({pred.confidence:.2f})'
