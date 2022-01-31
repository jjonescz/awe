import dataclasses

import pytorch_lightning as pl
import transformers
from transformers.models.big_bird.modeling_big_bird import \
    BigBirdForQuestionAnsweringModelOutput

import awe.qa.eval

ModelOutput = BigBirdForQuestionAnsweringModelOutput

@dataclasses.dataclass
class Prediction:
    batch: transformers.BatchEncoding
    outputs: ModelOutput

# pylint: disable=arguments-differ, unused-argument
class Model(pl.LightningModule):
    def __init__(self, model: transformers.BigBirdForQuestionAnswering):
        super().__init__()
        self.model = model
        self.evaluator = awe.qa.eval.ModelEvaluator()

    def configure_optimizers(self):
        return transformers.AdamW(self.parameters(), lr=1e-5)

    def forward(self, batch: transformers.BatchEncoding) -> ModelOutput:
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
            hp_metric = metrics['mean_acc']
            self.log('hp_metric', hp_metric, prog_bar=False)

        return prefixed

    def compute_metrics(self, batch: transformers.BatchEncoding):
        outputs = self.forward(batch)
        return self.evaluator.compute_metrics(Prediction(batch, outputs))

    def predict_step(self, batch, *_):
        outputs = self(batch)
        return Prediction(batch, outputs)
