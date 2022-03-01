import dataclasses
import random
import sys
import warnings
from typing import Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torch.utils.tensorboard
from tqdm.auto import tqdm

import awe.data.sampling
import awe.data.set.pages
import awe.data.set.swde
import awe.features.extraction
import awe.model.classifier
import awe.model.decoding
import awe.model.eval
import awe.training.context
import awe.training.logging
import awe.training.params


@dataclasses.dataclass
class RunInput:
    # Currently unused, but might be useful in the future.
    pages: list[awe.data.set.pages.Page]

    loader: torch.utils.data.DataLoader

    prefix: Optional[str] = None
    """
    Prefix used in TensorBoard. When `None`, logging to TensorBoard is disabled.
    """

    progress: Optional[Callable[[], tqdm]] = None

    progress_metrics: list[str] = ()

    progress_dict: Optional[dict[str]] = None

class Trainer:
    ds: awe.data.set.swde.Dataset = None
    label_map: awe.training.context.LabelMap
    extractor: awe.features.extraction.Extractor
    sampler: awe.data.sampling.Sampler
    train_pages: list[awe.data.set.pages.Page]
    val_pages: list[awe.data.set.pages.Page]
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    evaluator: awe.model.eval.Evaluator
    device: torch.device
    model: awe.model.classifier.Model
    version: awe.training.logging.Version
    writer: Optional[torch.utils.tensorboard.SummaryWriter] = None
    trainer: pl.Trainer
    optim: torch.optim.Optimizer
    train_progress: Optional[tqdm] = None
    val_progress: Optional[tqdm] = None
    step: int
    restored_state: Optional[dict[str]] = None

    def __init__(self,
        params: awe.training.params.Params,
        prev_trainer: Optional['Trainer'] = None
    ):
        # Preserve previously loaded data and pretrained models (enables faster
        # iteration during code changes and reloading in development).
        if prev_trainer is not None:
            difference = params.difference(prev_trainer.params)
            if difference:
                warnings.warn('Params of previous trainer differ from ' + \
                    f'provided params ({difference}).')
            print('Loading previous trainer.')
            for key, value in vars(prev_trainer).items():
                setattr(self, key, value)

        self.params = params

    def load_pretrained(self):
        pass

    def load_dataset(self):
        set_seed(42)

        state = None
        if self.ds is not None:
            # Preserve dataset state.
            state = self.ds.get_state()
            # Release memory used by previously-loaded dataset.
            self.ds = None
        self.ds = awe.data.set.swde.Dataset(
            suffix='-exact',
            only_verticals=('auto',),
            state=state
        )

    def prepare_features(self):
        """Splits data and prepares features on them."""

        set_seed(42)

        self.label_map = awe.training.context.LabelMap()
        self.extractor = awe.features.extraction.Extractor(self)
        self.sampler = awe.data.sampling.Sampler(self)

        # Load websites from one vertical.
        websites = self.ds.verticals[0].websites

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
        self.train_loader = self.create_dataloader(self.train_pages, 'train',
            shuffle=True,
            prepare=True,
        )
        self.val_loader = self.create_dataloader(self.val_pages, 'val')

    def create_dataloader(self,
        pages: list[awe.data.set.pages.Page],
        desc: str,
        shuffle: bool = False,
        prepare: bool = False
    ):
        set_seed(42)

        flags = {
            'shuffle': shuffle,
            'prepare': prepare
        }
        if any(flags.values()):
            desc = f'{desc} ({", ".join(k for k, v in flags.items() if v)})'

        return torch.utils.data.DataLoader(
            self.sampler.load(pages, desc=desc, prepare=prepare),
            batch_size=self.params.batch_size,
            collate_fn=awe.data.sampling.Collater(),
            shuffle=shuffle,
        )

    def create_run(self,
        pages: list[awe.data.set.pages.Page],
        desc: str,
        log: bool = False,
        shuffle: bool = False
    ):
        return RunInput(
            pages=pages,
            loader=self.create_dataloader(
                pages=pages,
                desc=desc,
                shuffle=shuffle
            ),
            prefix=desc if log else None,
        )

    def create_model(self):
        set_seed(42)

        self.evaluator = awe.model.eval.Evaluator(self)
        use_gpu = self.params.use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.model = awe.model.classifier.Model(self).to(self.device)

    def create_version(self):
        if self.params.restore_num is not None:
            self.version = awe.training.logging.Version(
                number=self.params.restore_num,
                name=self.params.version_name
            )
            print(self.restore_version(self.version))
        else:
            awe.training.logging.Version.delete_last(self.params.version_name)
            self.version = awe.training.logging.Version.create_new(
                self.params.version_name
            )
            self.restored_state = None

            # Save params.
            self.params.save_version(self.version)

        # Initialize TensorBoard logger.
        self.writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=self.version.version_dir_path,
        )

    def restore_version(self, version: awe.training.logging.Version):
        checkpoints = version.get_checkpoints()
        if len(checkpoints) == 0:
            raise RuntimeError(f'No checkpoints ({version.version_dir_path!r})')

        # Check params.
        restored_params = awe.training.params.Params.load_version(version)
        difference = self.params.difference(restored_params,
            ignore_vars=('restore_num',)
        )
        if difference:
            raise RuntimeError(
                f'Restored params differ from current params ({difference}).')

        return self.restore_checkpoint(checkpoints[-1])

    def restore_checkpoint(self, checkpoint: awe.training.logging.Checkpoint):
        print(f'Loading {checkpoint.file_path!r}...')
        self.restored_state = torch.load(checkpoint.file_path)
        print('Restoring model state...')
        return self.model.load_state_dict(self.restored_state['model'])

    def _reset_loop(self):
        set_seed(42)
        self.step = 0
        self.train_progress = None
        self.val_progress = None

    def _finalize(self):
        if self.writer is not None:
            self.writer.flush()
        if self.train_progress is not None:
            self.train_progress.close()
        if self.val_progress is not None:
            self.val_progress.close()
        self.optim = None

    def train(self,
        train_progress_metrics: list[str] = ('loss', 'f1/page'),
        val_progress_metrics: list[str] = ('loss', 'f1/page')
    ):
        self._reset_loop()
        self.optim = self.model.create_optimizer()

        if self.restored_state is not None:
            print('Restoring training state...')
            self.model.load_state_dict(self.restored_state['model'])
            self.optim.load_state_dict(self.restored_state['optim'])
            self.step = self.restored_state['step']
            start_epoch_idx = self.restored_state['epoch'] + 1
            best_val_loss = self.restored_state['best_val_loss']
        else:
            start_epoch_idx = 0
            best_val_loss = sys.maxsize

        train_run = RunInput(
            pages=self.train_pages,
            loader=self.train_loader,
            prefix='train',
            progress=lambda: self.train_progress,
            progress_metrics=train_progress_metrics
        )
        val_run = RunInput(
            pages=self.val_pages,
            loader=self.val_loader,
            prefix='val',
            progress=lambda: self.val_progress,
            progress_metrics=val_progress_metrics
        )
        for epoch_idx in tqdm(range(self.params.epochs), desc='train'):
            if epoch_idx < start_epoch_idx:
                # Skip over restored epochs in this way, so the progress bar is
                # in a proper state.
                continue

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
                state = {
                    'model': self.model.state_dict(),
                    'optim': self.optim.state_dict(),
                    'step': self.step,
                    'epoch': epoch_idx,
                    'best_val_loss': best_val_loss,
                }
                torch.save(state, ckpt.file_path)
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
                slow_eval.add_slow(awe.model.classifier.Prediction(batch, outputs))

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
            outputs = self.model.forward(batch)
            evaluation.add(awe.model.classifier.Prediction(batch, outputs))
            self.step += 1
            self.val_progress.update()
        return self._eval(run, evaluation)

    def _eval(self,
        run: RunInput,
        evaluation: awe.model.eval.Evaluation
    ):
        # Compute aggregate metrics.
        metrics = evaluation.compute()

        # Log to TensorBoard.
        if run.prefix is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f'{run.prefix}_{k}', v, self.step)

        # Update progress bar.
        if run.progress is not None and len(run.progress_metrics) != 0:
            progress_dict = {
                k: v
                for k in run.progress_metrics
                if (v := metrics.get(k)) is not None
            }

            # Preserve previous progress metrics.
            if run.progress_dict is not None:
                run.progress_dict.update(progress_dict)
                progress_dict = run.progress_dict
            else:
                run.progress_dict = progress_dict

            run.progress().set_postfix(progress_dict)

        return metrics

    def validate(self, run: RunInput):
        self._reset_loop()
        metrics = self._validate_epoch(run)
        self._finalize()
        return metrics

    def predict(self, run: RunInput):
        predictions: list[awe.model.classifier.Prediction] = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(run.loader, desc='pred'):
                outputs = self.model.forward(batch)
                predictions.append(awe.model.classifier.Prediction(batch, outputs))
        return predictions

    def decode(self, preds: list[awe.model.classifier.Prediction]):
        return awe.model.decoding.Decoder(self).decode(preds)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
