import dataclasses

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
import transformers
from transformers.models.big_bird.modeling_big_bird import \
    BigBirdForQuestionAnsweringModelOutput


@dataclasses.dataclass
class Prediction:
    batch: transformers.BatchEncoding
    outputs: BigBirdForQuestionAnsweringModelOutput

# pylint: disable=arguments-differ, unused-argument
class Model(pl.LightningModule):
    def __init__(self, model: transformers.BigBirdForQuestionAnswering):
        super().__init__()
        self.model = model
        self.metric = datasets.load_metric('accuracy')

    def configure_optimizers(self):
        return transformers.AdamW(self.parameters(), lr=1e-5)

    def forward(self,
        batch: transformers.BatchEncoding
    ) -> BigBirdForQuestionAnsweringModelOutput:
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            start_positions=batch['start_positions'],
            end_positions=batch['end_positions'],
        )

    def training_step(self, batch, *_):
        outputs = self.forward(batch)
        return outputs.loss

    def validation_step(self, batch, *_):
        return self._shared_eval_step('val', batch)

    def test_step(self, batch, *_):
        return self._shared_eval_step('test', batch)

    def _shared_eval_step(self, prefix: str, batch):
        metrics = self.compute_metrics(batch)
        prefixed = { f'{prefix}_{k}': v for k, v in metrics.items() }

        self.log_dict(prefixed)

        # Log `hp_metric` which is used as main metric in TensorBoard.
        if prefix == 'val':
            hp_metric = metrics['acc']
            self.log('hp_metric', hp_metric, prog_bar=False)

        return prefixed

    def compute_metrics(self, batch: transformers.BatchEncoding):
        outputs = self.forward(batch)
        loss = outputs.loss
        start_acc = self._compute_accuracy(
            logits=outputs.start_logits,
            labels=batch['start_positions'],
        )
        end_acc = self._compute_accuracy(
            logits=outputs.end_logits,
            labels=batch['end_positions'],
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

    def predict_step(self, batch, *_):
        outputs = self(batch)
        return Prediction(batch, outputs)