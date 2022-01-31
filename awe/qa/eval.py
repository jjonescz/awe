import datasets
import numpy as np
import torch

import awe.qa.model


class ModelEvaluator:
    def __init__(self):
        self.metric = datasets.load_metric('accuracy')

    def compute_metrics(self, pred: awe.qa.model.Prediction):
        loss = pred.outputs.loss
        start_acc = self._compute_accuracy(
            logits=pred.outputs.start_logits,
            labels=pred.batch['start_positions'],
        )
        end_acc = self._compute_accuracy(
            logits=pred.outputs.end_logits,
            labels=pred.batch['end_positions'],
        )
        return {
            'loss': loss,
            'start_acc': start_acc,
            'end_acc': end_acc,
            'acc': np.mean([start_acc, end_acc])
        }

    def _compute_accuracy(self, logits, labels):
        predictions = torch.argmax(logits, dim=-1)
        results = self.metric.compute(
            predictions=predictions,
            references=labels
        )
        return results['accuracy']
