import collections
import dataclasses
import json
import os
import sys
import warnings
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torch.utils.tensorboard
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
    save_every_n_epochs: Optional[int] = 1
    log_every_n_steps: int = 10
    eval_every_n_steps: Optional[int] = 50

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

@dataclasses.dataclass
class RunInput:
    loader: torch.utils.data.DataLoader

    prefix: Optional[str] = None
    """
    Prefix used in TensorBoard. When `None`, logging to TensorBoard is disabled.
    """

class Trainer:
    train_pages: list[awe_graph.HtmlPage]
    val_pages: list[awe_graph.HtmlPage]
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    model: awe.qa.model.Model
    version: awe.training.logging.Version
    writer: torch.utils.tensorboard.SummaryWriter
    trainer: pl.Trainer
    optim: torch.optim.Optimizer
    train_progress: Optional[tqdm] = None
    val_progress: Optional[tqdm] = None
    step: int

    def __init__(self, params: TrainerParams):
        self.params = params
        self.pipeline = awe.qa.pipeline.Pipeline()
        self.label_map = awe.qa.sampler.LabelMap()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.evaluator = awe.qa.eval.ModelEvaluator(self.label_map)
        self.running_loss = collections.defaultdict(float)
        self.metrics = collections.defaultdict(dict)

    def create_version(self):
        awe.training.logging.Version.delete_last(self.params.version_name)
        self.version = awe.training.logging.Version.create_new(
            self.params.version_name
        )

        # Save params.
        self.params.save_version(self.version)

        # Initialize TensorBoard logger.
        self.writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=self.version.version_dir_path,
        )

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
        self.model = awe.qa.model.Model(self.pipeline.model).to(self.device)

    def restore(self, checkpoint: awe.training.logging.Checkpoint):
        self.model.load_state_dict(torch.load(checkpoint.file_path))

    def _reset_loop(self):
        self.step = 0
        self.train_progress = None
        self.val_progress = None

    def _finalize(self):
        self.writer.flush()
        if self.train_progress is not None:
            self.train_progress.close()
        if self.val_progress is not None:
            self.val_progress.close()

    def train(self):
        self._reset_loop()
        self.running_loss.clear()
        self.optim = self.model.configure_optimizers()
        train_run = RunInput(self.train_loader, 'train')
        val_run = RunInput(self.val_loader, 'val')
        best_val_loss = sys.maxsize
        for epoch_idx in tqdm(range(self.params.epochs), desc='train'):
            train_metrics = self._train_epoch(train_run, val_run)
            val_metrics = self._validate_epoch(val_run)

            self.writer.add_scalar('epoch/per_step', epoch_idx, self.step)

            # Log per-epoch loss for comparison.
            for key, train_value in train_metrics.items():
                val_value = val_metrics[key]
                self.writer.add_scalar(f'epoch/{train_run.prefix}_{key}', train_value, self.step)
                self.writer.add_scalar(f'epoch/{val_run.prefix}_{key}', val_value, self.step)

            # Keep track of best validation loss.
            is_better_val_loss = val_metrics['loss'] < best_val_loss
            if is_better_val_loss:
                best_val_loss = val_metrics['loss']

            # Save model checkpoint if better loss reached or if Nth epoch.
            if is_better_val_loss or (
                self.params.save_every_n_epochs is not None and
                epoch_idx % self.params.save_every_n_epochs == 0
            ):
                ckpt = self.version.create_checkpoint(epoch=epoch_idx, step=self.step)
                torch.save(self.model.state_dict(), ckpt.file_path)
        self._finalize()

    def _train_epoch(self, run: RunInput, val_run: RunInput):
        self.model.train()
        if self.train_progress is None:
            self.train_progress = tqdm(desc='epoch')
        self.train_progress.reset(total=len(run.loader))
        fast_eval = self.evaluator.start_evaluation() # cleared every Nth step
        total_eval = self.evaluator.start_evaluation() # collected every step
        slow_eval = self.evaluator.start_evaluation() # collected every Nth step
        for batch_idx, batch in enumerate(run.loader):
            self.optim.zero_grad()
            batch = batch.to(self.device)
            outputs = self.model.forward(batch)
            fast_eval.add_fast(outputs)
            total_eval.add_fast(outputs)
            outputs.loss.backward()
            self.optim.step()
            self.step += 1
            self.train_progress.update()

            if batch_idx % self.params.log_every_n_steps == 0:
                # Log aggregate train loss.
                self._eval(run, fast_eval)
                fast_eval.clear()

                # Compute all metrics every once in a while.
                slow_eval.add_slow(awe.qa.model.Prediction(batch, outputs))

            # Validate during training.
            if (self.params.eval_every_n_steps is not None and
                batch_idx % self.params.eval_every_n_steps == 0):
                self._validate_epoch(val_run)
                self.model.train()

        # Log aggregate metrics.
        self._eval(run, fast_eval)
        self._eval(run, slow_eval)
        return total_eval.compute()

    def _validate_epoch(self, run: RunInput):
        with torch.no_grad():
            return self._validate_epoch_core(run)

    def _validate_epoch_core(self, run: RunInput):
        self.model.eval()
        if self.val_progress is None:
            self.val_progress = tqdm(desc='val')
        self.val_progress.reset(total=len(run.loader))
        evaluation = self.evaluator.start_evaluation()
        for batch in run.loader:
            batch = batch.to(self.device)
            outputs = self.model.forward(batch)
            evaluation.add(awe.qa.model.Prediction(batch, outputs))
            self.step += 1
            self.val_progress.update()
        return self._eval(run, evaluation)

    def _eval(self, run: RunInput, evaluation: awe.qa.eval.ModelEvaluation):
        # Log aggregate metrics to TensorBoard.
        metrics = evaluation.compute()
        if run.prefix is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f'{run.prefix}_{k}', v, self.step)
        return metrics

    def validate(self, run: RunInput):
        self._reset_loop()
        metrics = self._validate_epoch(run)
        self._finalize()
        return metrics

    def predict(self, run: RunInput):
        predictions: list[awe.qa.model.Prediction] = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(run.loader, desc='predict'):
                batch = batch.to(self.device)
                outputs = self.model.forward(batch)
                predictions.append(awe.qa.model.Prediction(batch, outputs))
        return predictions

    def decode(self, preds: list[awe.qa.model.Prediction]):
        decoder = awe.qa.decoder.Decoder(self.pipeline.tokenizer)
        return decoder.decode_predictions(preds)
