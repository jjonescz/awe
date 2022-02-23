import collections
from typing import TYPE_CHECKING

import torchmetrics

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

class Evaluation:
    metrics: dict[str, FloatMetric]

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.metrics = collections.defaultdict(FloatMetric)
        self.nodes = Metrics(evaluator, [
            torchmetrics.Accuracy(ignore_index=0),
            torchmetrics.F1(ignore_index=0)
        ])

    def clear(self):
        self.metrics.clear()

    def compute(self):
        metrics_dict = { k: v.compute() for k, v in self.metrics.items() }
        return metrics_dict | self.nodes.compute()

    def add(self, pred: 'awe.model.classifier.Prediction'):
        self.add_fast(pred.outputs)
        self.add_slow(pred)

    def add_fast(self, outputs: 'awe.model.classifier.ModelOutput'):
        self.metrics['loss'].add(outputs.loss.item())

    def add_slow(self, pred: 'awe.model.classifier.Prediction'):
        self.nodes.update(preds=pred.outputs.logits, target=pred.outputs.gold_labels)
