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

class Evaluation:
    metrics: dict[str, FloatMetric]
    total_stats_updated: bool = False

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.metrics = collections.defaultdict(FloatMetric)
        self.total_stats = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(),
            torchmetrics.F1()
        ]).to(evaluator.trainer.device)

    def clear(self):
        self.metrics.clear()

    def compute(self):
        metrics_dict = { k: v.compute() for k, v in self.metrics.items() }
        total_stats_dict = self.total_stats.compute() if self.total_stats_updated else {}
        total_stats_dict = { k: v.item() for k, v in total_stats_dict.items() }
        return metrics_dict | total_stats_dict

    def add(self, pred: 'awe.model.classifier.Prediction'):
        self.add_fast(pred.outputs)
        self.add_slow(pred)

    def add_fast(self, outputs: 'awe.model.classifier.ModelOutput'):
        self.metrics['loss'].add(outputs.loss.item())

    def add_slow(self, pred: 'awe.model.classifier.Prediction'):
        self.total_stats.update(preds=pred.outputs.logits, target=pred.outputs.gold_labels)
        self.total_stats_updated = True
