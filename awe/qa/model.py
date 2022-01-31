import dataclasses
from typing import TYPE_CHECKING

import pytorch_lightning as pl
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

# pylint: disable=arguments-differ, unused-argument
class Model(pl.LightningModule):
    def __init__(self,
        model: transformers.BigBirdForQuestionAnswering,
        params: 'awe.qa.trainer.TrainerParams',
        evaluator: awe.qa.eval.ModelEvaluator,
    ):
        super().__init__()
        self.model = model
        self.params = params
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

    def training_step(self, batch, batch_idx: int, *_):
        outputs = self.forward(batch)
        loss = outputs.loss

        if batch_idx % self.params.log_every_n_steps == 0:
            self._shared_eval_step('train', batch)
        else:
            self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, *_):
        return self._shared_eval_step('val', batch)

    def test_step(self, batch, *_):
        return self._shared_eval_step('test', batch)

    def _shared_eval_step(self, prefix: str, batch):
        metrics = self.compute_metrics(batch)
        prefixed = { f'{prefix}_{k}': v for k, v in metrics.items() }

        is_val = prefix == 'val'
        self.log_dict(prefixed, prog_bar=is_val)

        # Log `hp_metric` which is used as main metric in TensorBoard.
        if is_val:
            hp_metric = metrics['mean_post_acc']
            self.log('hp_metric', hp_metric, prog_bar=False)

        return prefixed

    def compute_metrics(self, batch: transformers.BatchEncoding):
        outputs = self.forward(batch)
        return self.evaluator.compute_metrics(Prediction(batch, outputs))

    def predict_step(self, batch, *_):
        outputs = self(batch)
        return Prediction(batch, outputs)
