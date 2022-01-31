import datasets
import numpy as np
import torch

import awe.qa.model
import awe.qa.postprocess
import awe.qa.sampler
import awe.utils


class ModelEvaluator:
    def __init__(self, label_map: awe.qa.sampler.LabelMap):
        self.label_map = label_map
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

        # Compute match per label.
        def compute_per_label(label: str):
            matches = [
                start_pred == start_gold and end_pred == end_gold
                for start_pred, start_gold, end_pred, end_gold, label_id in zip(
                    start_preds, pred.batch['start_positions'],
                    end_preds, pred.batch['end_positions'],
                    pred.batch['label_ids']
                )
                if self.label_map.label_to_id[label] == label_id
            ]
            return sum(matches) / len(matches)
        matches = {
            f'{label}_acc': compute_per_label(label)
            for label in self.label_map.label_to_id.keys()
        }

        return {
            f'start_{prefix}_acc': start_acc,
            f'end_{prefix}_acc': end_acc,
            f'mean_{prefix}_acc': np.mean([start_acc, end_acc]),
            **matches,
        }
