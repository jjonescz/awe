import dataclasses
from typing import TYPE_CHECKING

import torch
import transformers
from transformers.models.big_bird.modeling_big_bird import \
    BigBirdForQuestionAnsweringModelOutput

import awe.qa.eval

if TYPE_CHECKING:
    import awe.qa.trainer

ModelOutput = BigBirdForQuestionAnsweringModelOutput

@dataclasses.dataclass
class Prediction:
    batch: transformers.BatchEncoding
    outputs: ModelOutput

class Model(torch.nn.Module):
    def __init__(self,
        model: transformers.BigBirdForQuestionAnswering,
        trainer: 'awe.qa.trainer.Trainer',
        evaluator: awe.qa.eval.ModelEvaluator,
    ):
        super().__init__()
        self.model = model
        self.trainer = trainer
        self.evaluator = evaluator

    def configure_optimizers(self):
        return transformers.AdamW(self.parameters(), lr=1e-5)

    def forward(self, batch: transformers.BatchEncoding) -> ModelOutput:
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            start_positions=batch['start_positions'],
            end_positions=batch['end_positions'],
        )

    def training_step(self, batch, batch_idx: int):
        outputs = self.forward(batch)
        loss = outputs.loss

        if batch_idx % self.trainer.params.log_every_n_steps == 0:
            self._shared_eval_step('train', batch)

        return loss

    def validation_step(self, batch, batch_idx: int):
        return self._shared_eval_step('val', batch)

    def test_step(self, batch, batch_idx: int):
        return self._shared_eval_step('test', batch)

    def _shared_eval_step(self, prefix: str, batch):
        metrics = self.compute_metrics(batch)

        for k, v in metrics.items():
            self.trainer.metrics[prefix][k] = v

        return metrics

    def compute_metrics(self, batch: transformers.BatchEncoding):
        outputs = self.forward(batch)
        return self.evaluator.compute_metrics(Prediction(batch, outputs))

    def predict_step(self, batch):
        outputs = self(batch)
        return Prediction(batch, outputs)
