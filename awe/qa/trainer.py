import dataclasses
import json
import os
import warnings
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torch.utils.tensorboard
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from tqdm.auto import tqdm

import awe.qa.collater
import awe.qa.decoder
import awe.qa.eval
import awe.qa.model
import awe.qa.pipeline
import awe.qa.sampler
import awe.training.callbacks
import awe.training.logging
from awe import awe_graph
from awe.data import constants, swde

#  Ignore warnings.
warnings.filterwarnings('ignore', message='__floordiv__ is deprecated')

@dataclasses.dataclass
class TrainerParams:
    train_subset: int = 2000
    val_subset: int = 50
    epochs: int = 5
    version_name: str = ''
    batch_size: int = 16
    max_length: Optional[int] = None
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 50

    @classmethod
    def load_version(cls, version: awe.training.logging.Version):
        return cls.load_file(version.params_path)

    @classmethod
    def load_user(cls):
        """Loads params from user-provided file."""
        path = f'{constants.DATA_DIR}/qa-params.json'
        if not os.path.exists(path):
            # Create file with default params as template.
            warnings.warn(f'No params file, creating one at {repr(path)}.')
            TrainerParams().save_file(path)
            return None
        return cls.load_file(path)

    @staticmethod
    def load_file(path: str):
        with open(path, mode='r', encoding='utf-8') as f:
            return TrainerParams(**json.load(f))

    def save_version(self, version: awe.training.logging.Version):
        self.save_file(version.params_path)

    def save_file(self, path: str):
        with open(path, mode='w', encoding='utf-8') as f:
            json.dump(dataclasses.asdict(self), f,
                indent=2,
                sort_keys=True
            )

    def update_from(self, checkpoint: awe.training.logging.Checkpoint):
        self.epochs = checkpoint.epoch + 1

class Trainer:
    train_pages: list[awe_graph.HtmlPage]
    val_pages: list[awe_graph.HtmlPage]
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    model: awe.qa.model.Model
    version: awe.training.logging.Version
    trainer: pl.Trainer
    optim: torch.optim.Optimizer
    metrics: dict[str, dict[str, float]]

    def __init__(self, params: TrainerParams):
        self.params = params
        self.pipeline = awe.qa.pipeline.Pipeline()
        self.label_map = awe.qa.sampler.LabelMap()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.metrics = {}

    def create_version(self):
        awe.training.logging.Version.delete_last(self.params.version_name)
        self.version = awe.training.logging.Version.create_new(
            self.params.version_name
        )

        # Save params.
        self.params.save_version(self.version)

    def load_pipeline(self):
        self.pipeline.load()

    def load_data(self):
        # Load websites from one vertical.
        sds = swde.Dataset(suffix='-exact')
        websites = sds.verticals[0].websites

        # Split websites.
        train_website_indices = [0, 3, 4, 5, 7]
        val_website_indices = [i
            for i in range(len(websites))
            if i not in train_website_indices
        ]
        train_websites = [websites[i] for i in train_website_indices]
        val_websites = [websites[i] for i in val_website_indices]
        train_website_names = [w.name for w in train_websites]
        val_website_names = [w.name for w in val_websites]
        print(f'{train_website_names=}, {val_website_names=}')

        # Take pages.
        train_pages = [p for w in train_websites for p in w.pages]
        val_pages = [p for w in val_websites for p in w.pages]
        print(f'{len(train_pages)=}, {len(val_pages)=}')

        # Take subset.
        rng = np.random.default_rng(42)
        self.train_pages = rng.choice(train_pages, self.params.train_subset, replace=False)
        self.val_pages = rng.choice(val_pages, self.params.val_subset, replace=False)
        print(f'{len(self.train_pages)=}, {len(self.val_pages)=}')

        # Create dataloaders.
        self.train_loader = self.create_dataloader(self.train_pages, shuffle=True)
        self.val_loader = self.create_dataloader(self.val_pages)

    def create_dataloader(self,
        pages: list[awe_graph.HtmlPage],
        shuffle: bool = False
    ):
        samples = awe.qa.sampler.get_samples(pages)
        return torch.utils.data.DataLoader(
            samples,
            batch_size=self.params.batch_size,
            collate_fn=awe.qa.collater.Collater(
                self.pipeline.tokenizer,
                self.label_map,
                max_length=self.params.max_length,
            ),
            shuffle=shuffle,
        )

    def create_model(self):
        self.model = awe.qa.model.Model(
            self.pipeline.model,
            self,
            awe.qa.eval.ModelEvaluator(self.label_map),
        ).to(self.device)

    def train(self):
        self.writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=self.version.version_dir_path,
        )
        self.step = 0
        self.metrics['train'] = {
            'running_loss': 0.0,
        }
        self.metrics['val'] = {
            'running_loss': 0.0,
        }
        self.optim = self.model.configure_optimizers()
        for epoch_idx in tqdm(range(self.params.epochs), desc='train'):
            avg_train_loss = self._train_epoch(epoch_idx)
            avg_val_loss = self._validate_epoch(epoch_idx)

            # Log metrics.
            self.writer.add_scalar('epoch/train/loss', avg_train_loss, epoch_idx)
            self.writer.add_scalar('epoch/val/loss', avg_val_loss, epoch_idx)
            self.writer.flush()

    def _train_epoch(self, epoch_idx: int):
        self.model.train()
        self.train_progress = tqdm(self.train_loader, desc='epoch')
        for batch_idx, batch in enumerate(self.train_progress):
            self._train_batch(batch, batch_idx, epoch_idx)
            self.step += 1
        return self._train_loss(len(self.train_loader) - 1, epoch_idx)

    def _validate_epoch(self, epoch_idx: int):
        self.model.eval()
        self.val_progress = tqdm(self.val_loader, desc='val')
        for batch_idx, batch in enumerate(self.val_progress):
            self._validate_batch(batch, batch_idx, epoch_idx)
            self.step += 1
        return self.metrics['val']['running_loss'] / len(self.val_loader)

    def _train_batch(self, batch, batch_idx: int, epoch_idx: int):
        batch = batch.to(self.device)
        self.optim.zero_grad()
        loss = self.model.training_step(batch, batch_idx)
        loss.backward()
        self.optim.step()

        # Gather metrics.
        self.metrics['train']['running_loss'] += loss.item()
        if batch_idx % self.params.log_every_n_steps == 0:
            self._train_loss(batch_idx, epoch_idx)

        # TODO: Validate during training.
        if batch_idx % self.params.eval_every_n_steps == 0:
            pass

    def _train_loss(self, batch_idx: int, epoch_idx: int):
        last_loss = self.metrics['train']['running_loss'] / self.params.log_every_n_steps
        self.writer.add_scalar('train/loss', last_loss, self.step)
        self.metrics['train']['running_loss'] = 0.0
        self.train_progress.set_postfix({ 'loss': last_loss })
        return last_loss

    def _validate_batch(self, batch, batch_idx: int, epoch_idx: int):
        batch = batch.to(self.device)
        metrics = self.model.validation_step(batch, batch_idx)
        self.metrics['val']['running_loss'] += metrics['loss']

    def restore(self, checkpoint: awe.training.logging.Checkpoint):
        self.params.update_from(checkpoint)
        self._create_trainer(
            resume_from_checkpoint=checkpoint.file_path
        )
        self.trainer.fit(self.model, self.train_loader)

    def _create_trainer(self,
        logger: Optional[TensorBoardLogger] = None,
        resume_from_checkpoint: Optional[str] = None
    ):
        self.trainer = pl.Trainer(
            gpus=torch.cuda.device_count(),
            max_epochs=self.params.epochs,
            logger=logger,
            resume_from_checkpoint=resume_from_checkpoint,
            callbacks=[awe.training.callbacks.EvalDuringTraining(self)],
        )

    def validate(self, pages: list[awe_graph.HtmlPage]):
        loader = self.create_dataloader(pages)
        return self.trainer.validate(self.model, loader)

    def predict(self, pages: list[awe_graph.HtmlPage]):
        loader = self.create_dataloader(pages)
        return self.trainer.predict(self.model, loader)

    def decode(self, preds: list[awe.qa.model.Prediction]):
        decoder = awe.qa.decoder.Decoder(self.pipeline.tokenizer)
        return decoder.decode_predictions(preds)
