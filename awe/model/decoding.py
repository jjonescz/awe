from typing import TYPE_CHECKING

import pandas as pd

import awe.data.graph.dom
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

    def decode(self, preds: list[awe.model.classifier.Prediction]):
        # Get all pages in preds.
        pages = {
            node.dom.page
            for pred in preds
            for node in pred.batch
        }

        return pd.DataFrame(
            self.decode_one(page)
            for page in pages
        )

    def decode_one(self, page: awe.data.set.pages.Page):
        d = {
            # 'vertical': page.website.vertical.name,
            # 'website': page.website.name,
            # 'index': page.index,
            'url': page.url
        }

        page_dom = page.try_get_dom()
        for label_key, labeled_nodes in page_dom.labeled_nodes.items():
            d[f'gold_{label_key}'] = [self.decode_node(n) for n in labeled_nodes]

            # Sort by most confident predictions.
            pred_nodes = sorted(
                page_dom.node_predictions.get(label_key, ()),
                key=lambda p: p.confidence,
                reverse=True,
            )

            d[f'pred_{label_key}'] = [self.decode_node_pred(n) for n in pred_nodes]

        return d

    def decode_node(self, node: awe.data.graph.dom.Node):
        return node.text if node.is_text else f'<{node.html_tag}>'

    def decode_node_pred(self, pred: awe.data.graph.dom.NodePrediction):
        return f'{self.decode_node(pred.node)}[{pred.confidence:.2f}]'
