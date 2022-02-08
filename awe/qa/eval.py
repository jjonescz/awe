import collections
from typing import TYPE_CHECKING

import datasets
import numpy as np
import torch

import awe.qa.postprocess
import awe.qa.sampler
import awe.utils

if TYPE_CHECKING:
    import awe.qa.model


class ModelEvaluator:
    def __init__(self, label_map: awe.qa.sampler.LabelMap):
        self.label_map = label_map
        self.metric = datasets.load_metric('accuracy')

    def start_evaluation(self):
        return ModelEvaluation(self)

class FloatMetric:
    total: float = 0.0
    count: int = 0

    def add(self, value: float):
        self.total += value
        self.count += 1

    def compute(self):
        return self.total / self.count

class ModelEvaluation:
    metrics: dict[str, FloatMetric]

    def __init__(self, evaluator: ModelEvaluator):
        self.evaluator = evaluator
        self.metrics = collections.defaultdict(FloatMetric)

    def clear(self):
        self.metrics.clear()

    def compute(self):
        return { k: v.compute() for k, v in self.metrics.items() }

    def add_fast(self, outputs: 'awe.qa.model.ModelOutput'):
        self.metrics['loss'].add(outputs.loss.item())

    def add(self, pred: 'awe.qa.model.Prediction'):
        self.add_fast(pred.outputs)
        self._add_accuracies(pred, 'pre')
        self._add_accuracies(pred, 'post', clamp=True)

    def _add_accuracies(self,
        pred: 'awe.qa.model.Prediction',
        prefix: str,
        clamp: bool = False
    ):
        start_preds = torch.argmax(pred.outputs.start_logits, dim=-1)
        end_preds = torch.argmax(pred.outputs.end_logits, dim=-1)

        if clamp:
            clamped = awe.qa.postprocess.clamp_spans(
                start_preds, end_preds, pred.batch
            )
            start_preds, end_preds = awe.utils.unzip(clamped)

        start_acc = self.evaluator.metric.compute(
            predictions=start_preds,
            references=pred.batch['start_positions']
        )['accuracy']
        end_acc = self.evaluator.metric.compute(
            predictions=end_preds,
            references=pred.batch['end_positions']
        )['accuracy']
        self.metrics[f'{prefix}_acc/start'].add(start_acc)
        self.metrics[f'{prefix}_acc/end'].add(end_acc)
        self.metrics[f'{prefix}_acc/mean'].add(np.mean([start_acc, end_acc]))

        # Compute match per label.
        def compute_per_label(target_label: str, target_label_id: int):
            matches = [
                start_pred == start_gold and end_pred == end_gold
                for start_pred, start_gold, end_pred, end_gold, label_id in zip(
                    start_preds, pred.batch['start_positions'],
                    end_preds, pred.batch['end_positions'],
                    pred.batch['label_ids']
                )
                if target_label_id == label_id
            ]
            if len(matches) == 0:
                return {}
            return {
                f'{prefix}_label_acc/{target_label}': sum(matches) / len(matches)
            }
        label_matches = {
            k: v
            for label_id, label in self.evaluator.label_map.id_to_label.items()
            for k, v in compute_per_label(label, label_id).items()
        }
        for k, v in label_matches.items():
            self.metrics[k].add(v.item())

        # Add total label accuracy.
        label_accuracies = torch.FloatTensor(list(label_matches.values()))
        total_acc = torch.mean(label_accuracies).item()
        self.metrics[f'{prefix}_label_acc/total'].add(total_acc)
