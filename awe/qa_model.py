import datasets
import pytorch_lightning as pl
import torch
import transformers


# pylint: disable=arguments-differ, unused-argument
class QaModel(pl.LightningModule):
    def __init__(self,
        model_id = 'vasudevgupta/bigbird-roberta-natural-questions'
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model_id = model_id
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_id)
        self.metric = datasets.load_metric('accuracy')

    def configure_optimizers(self):
        return transformers.AdamW(self.parameters(), lr=1e-5)

    def create_tokenizer(self):
        return transformers.AutoTokenizer.from_pretrained(self.model_id)

    def forward(self, encodings):
        return self.model(**encodings)

    def training_step(self, batch, *_):
        outputs = self.forward(batch)
        return outputs.loss

    def validation_step(self, batch, *_):
        return self._shared_eval_step(self, batch)

    def test_step(self, batch, *_):
        return self._shared_eval_step(self, batch)

    def _shared_eval_step(self, prefix: str, batch):
        outputs = self.forward(batch)
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        accuracy = self.metric.compute(
            predictions=predictions,
            references=batch['labels']
        )

        results = {
            'loss': loss,
            'acc': accuracy,
        }
        prefixed = { f'{prefix}_{k}': v for k, v in results.items() }

        self.log_dict(prefixed)

        # Log `hp_metric` which is used as main metric in TensorBoard.
        if prefix == 'val':
            self.log('hp_metric', accuracy, prog_bar=False)

        return prefixed
