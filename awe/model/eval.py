import collections
from typing import TYPE_CHECKING

import awe.training.context
import awe.utils

if TYPE_CHECKING:
    import awe.model.classifier


class Evaluator:
    def __init__(self, label_map: awe.training.context.LabelMap):
        self.label_map = label_map

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

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.metrics = collections.defaultdict(FloatMetric)

    def clear(self):
        self.metrics.clear()

    def compute(self):
        return { k: v.compute() for k, v in self.metrics.items() }

    def add(self, pred: 'awe.model.classifier.Prediction'):
        self.add_fast(pred.outputs)
        self.add_slow(pred)

    def add_fast(self, outputs: 'awe.model.classifier.ModelOutput'):
        self.metrics['loss'].add(outputs.loss.item())

    def add_slow(self, pred: 'awe.model.classifier.Prediction'):
        pass
