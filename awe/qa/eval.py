import datasets
import numpy as np
import torch

import awe.qa.model
import awe.qa.postprocess
import awe.utils


class ModelEvaluator:
    def __init__(self):
        self.metric = datasets.load_metric('accuracy')

    def compute_metrics(self, pred: awe.qa.model.Prediction):
        loss = pred.outputs.loss
        return {
            'loss': loss,
            **self._compute_accuracies(pred, 'pre'),
            **self._compute_accuracies(pred, 'post', clamp=True),
        }

    def _compute_accuracies(self,
        pred: awe.qa.model.Prediction,
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

        start_acc = self.metric.compute(
            predictions=start_preds,
            references=pred.batch['start_positions']
        )['accuracy']
        end_acc = self.metric.compute(
            predictions=end_preds,
            references=pred.batch['end_positions']
        )['accuracy']

        return {
            f'start_{prefix}_acc': start_acc,
            f'end_{prefix}_acc': end_acc,
            f'mean_{prefix}_acc': np.mean([start_acc, end_acc])
        }
