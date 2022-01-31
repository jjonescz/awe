from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch

if TYPE_CHECKING:
    import awe.qa.trainer


class EvalDuringTraining(pl.Callback):
    def __init__(self, trainer: 'awe.qa.trainer.Trainer'):
        self.trainer = trainer

    def on_train_batch_start(self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
        dataloader_idx: int
    ):
        if batch_idx % self.trainer.params.eval_every_n_steps == 0:
            with torch.no_grad():
                self.trainer.validate(self.trainer.val_pages)
        return super().on_train_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )
