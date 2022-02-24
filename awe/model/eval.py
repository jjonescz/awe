import collections
from typing import TYPE_CHECKING

import torchmetrics

import awe.data.graph.pred
import awe.data.set.pages
import awe.model.metrics
import awe.utils

if TYPE_CHECKING:
    import awe.model.classifier
    import awe.training.trainer


class Evaluator:
    def __init__(self, trainer: 'awe.training.trainer.Trainer'):
        self.trainer = trainer

    def start_evaluation(self):
        return Evaluation(self)

class FloatMetric:
    total: float = 0.0
    count: int = 0

    def add(self, value: float):
        self.total += value
        self.count += 1

    def compute(self):
        return self.total / self.count

class Metrics:
    """Wrapper for `MetricCollection` handling some edge cases."""

    updated: bool = False

    def __init__(self, evaluator: Evaluator, *args, **kwargs):
        self.collection = torchmetrics.MetricCollection(*args, **kwargs) \
            .to(evaluator.trainer.device)

    def update(self, *args, **kwargs):
        self.collection.update(*args, **kwargs)
        self.updated = True

    def compute(self):
        d = self.collection.compute() if self.updated else {}
        return { k: v.item() for k, v in d.items() }

    def reset(self):
        self.collection.reset()
        self.updated = False

class Evaluation:
    metrics: dict[str, FloatMetric]

    pred_set_dirty: bool = False

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.pred_set = awe.data.graph.pred.PredictionSet(evaluator.trainer)
        self.metrics = collections.defaultdict(FloatMetric)
        self.nodes = Metrics(evaluator, {
            'acc': torchmetrics.Accuracy(ignore_index=0),
            'f1': torchmetrics.F1(ignore_index=0)
        }, postfix='/node')

    def clear(self):
        self.pred_set.clear()
        self.metrics.clear()
        self.nodes.reset()

    def compute(self):
        metrics_dict = { k: v.compute() for k, v in self.metrics.items() }
        metrics_dict.update(self.nodes.compute())

        # Compute page-wide metrics.
        if self.pred_set_dirty:
            self.pred_set_dirty = False
            per_label = {
                label_key: awe.model.metrics.PredStats() for label_key
                in self.evaluator.trainer.label_map.label_to_id.keys()
            }
            for page, pred_page in self.pred_set.preds.items():
                # Skip pages that haven't been predicted yet.
                if pred_page.num_predicted == 0:
                    continue

                for label_key, pred_list in pred_page.preds.items():
                    stats = per_label[label_key]
                    gold = page.dom.labeled_nodes.get(label_key)
                    if not gold:
                        # Negative sample is when no node is labeled.
                        if not pred_list:
                            stats.true_negatives += 1
                        else:
                            stats.false_negatives += 1
                    else:
                        # Find most confident prediction.
                        best_pred = awe.utils.where_max(pred_list,
                            lambda p: p.confidence).node
                        if best_pred in gold:
                            stats.true_positives += 1
                        else:
                            stats.false_positives += 1

            # Log per-label stats.
            per_label_metrics = {
                label_key: awe.model.metrics.F1Metrics.compute(stats)
                for label_key, stats in per_label.items()
            }
            for k, m in per_label_metrics.items():
                metrics_dict.update(m.to_dict(postfix=f'/label_{k}'))

            # Average per-label stats to page-wide stats.
            page_metrics = awe.model.metrics.F1Metrics.from_vector(sum(
                metrics.to_vector() for metrics in per_label_metrics.values())
                / len(per_label))

            metrics_dict.update(page_metrics.to_dict(postfix='/page'))

        return metrics_dict

    def add(self, pred: 'awe.model.classifier.Prediction'):
        self.add_fast(pred.outputs)
        self.add_slow(pred)

    def add_fast(self, outputs: 'awe.model.classifier.ModelOutput'):
        self.metrics['loss'].add(outputs.loss.item())

    def add_slow(self, pred: 'awe.model.classifier.Prediction'):
        self.nodes.update(preds=pred.outputs.logits, target=pred.outputs.gold_labels)
        self.pred_set.add_batch(pred)
        self.pred_set_dirty = True
