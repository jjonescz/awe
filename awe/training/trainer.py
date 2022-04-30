"""Main entrypoint for training."""

import collections
import dataclasses
import itertools
import json
import os
import random
import sys
import warnings
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.utils.tensorboard
import torchinfo
from tqdm.auto import tqdm

import awe.data.graph.dom
import awe.data.sampling
import awe.data.set.apify
import awe.data.set.pages
import awe.data.set.swde
import awe.features.extraction
import awe.features.text
import awe.features.visual
import awe.model.classifier
import awe.model.decoding
import awe.model.eval
import awe.training.context
import awe.training.params
import awe.training.versioning


@dataclasses.dataclass
class RunInput:
    """Input for training, evaluation, or prediction."""

    # Currently unused, but might be useful in the future.
    pages: list[awe.data.set.pages.Page]
    """Set of pages used to create the set of sampled nodes."""

    loader: torch.utils.data.DataLoader
    """Loader producing the sampled nodes."""

    name: str
    """Name used for reporting (e.g., `train`, `val`, `test`)."""

    prefix: Optional[str] = None
    """
    Prefix used in TensorBoard. When `None`, logging to TensorBoard is disabled.
    """

    progress: Optional[Callable[[], tqdm]] = None
    """Progress bar unless disabled."""

    progress_metrics: list[str] = ()
    """Metric names to show in the `progress` bar."""

    progress_dict: Optional[dict[str]] = None
    """Current `progress_metrics` and their values."""

    lazy: bool = False
    """
    Whether pages should be prepared lazily. Otherwise, all pages are prepared
    up-front and the sampled nodes are cached in a list.
    """

class Subsetter:
    """Takes random subset of pages."""

    def __init__(self):
        self.rng = np.random.default_rng(42)

    def __call__(self,
        websites: list[awe.data.set.pages.Website],
        subset: Optional[int]
    ):
        if subset is None:
            return [p for w in websites for p in w.pages]
        return [
            p
            for w in websites
            for p in self._subset_one(website=w, subset=subset)
        ]

    def _subset_one(self, website: awe.data.set.pages.Website, subset: int):
        if len(website.pages) <= subset:
            warnings.warn(
                f'Cannot subset {subset} from website {website.name!r}, ' +
                f'taking all pages ({len(website.pages)}).')
            return website.pages
        return self.rng.choice(website.pages, subset, replace=False)

class Trainer:
    """Calls data loading code. Contains training and validation loops."""

    ds: awe.data.set.pages.Dataset = None
    label_map: awe.training.context.LabelMap
    extractor: awe.features.extraction.Extractor
    vertical: awe.data.set.pages.Vertical
    train_websites: list[awe.data.set.pages.Website]
    val_websites: list[awe.data.set.pages.Website]
    train_pages: list[awe.data.set.pages.Page]
    val_pages: list[awe.data.set.pages.Page]
    test_pages: list[awe.data.set.pages.Page]
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    evaluator: awe.model.eval.Evaluator
    device: torch.device
    model: awe.model.classifier.Model
    version: awe.training.versioning.Version
    writer: Optional[torch.utils.tensorboard.SummaryWriter] = None
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
                warnings.warn('Params of previous trainer differ from ' +
                    f'provided params ({difference}).')
            print('Loading previous trainer.')
            for key, value in vars(prev_trainer).items():
                setattr(self, key, value)

        self.params = params

    def load_dataset(self):
        """Loads dataset of pages."""

        set_seed(42)

        if self.params.dataset == awe.training.params.Dataset.swde:
            self.ds = awe.data.set.swde.Dataset(
                suffix='-exact',
                only_verticals=(self.params.vertical,)
            )
        elif self.params.dataset == awe.training.params.Dataset.apify:
            self.ds = awe.data.set.apify.Dataset(
                exclude_websites=self.params.exclude_websites,
                only_label_keys=self.params.label_keys
            )
        else:
            raise ValueError(
                f'Unrecognized dataset param {self.params.dataset!r}.')

    def init_features(self):
        """
        Creates fresh instances of features.

        Next, these blank features should be populated with context (such as the
        set of possible HTML tags) by preparing them on training data (via
        method `create_dataloader`) or loading from checkpoint (via method
        `restore_features`).
        """

        set_seed(42)

        # Create device (some features need it when preparing tensors).
        if self.params.use_gpu:
            if torch.cuda.is_available():
                use_gpu = True
            else:
                warnings.warn('GPU not available, falling back to CPU.')
                use_gpu = False
        else:
            use_gpu = False
        self.device = torch.device('cuda:0' if use_gpu else 'cpu')

        self.label_map = awe.training.context.LabelMap()
        self.extractor = awe.features.extraction.Extractor(self)

    def split_data(self):
        """Creates subsets of pages for training, validation, and testing."""

        set_seed(42)

        # Load websites from one vertical.
        self.vertical = self.ds.verticals[0]
        websites = self.vertical.websites

        if len(websites) == 0:
            raise RuntimeError('No websites found in dataset.')

        # Split websites.
        train_website_indices = self.params.train_website_indices
        val_website_indices = [i
            for i in range(len(websites))
            if i not in train_website_indices
        ]
        if len(val_website_indices) == 0:
            val_website_indices = train_website_indices
            warnings.warn(
                'No unseen websites left, using all websites for validation')
        self.train_websites = [websites[i] for i in train_website_indices]
        self.val_websites = [websites[i] for i in val_website_indices]
        train_website_names = [w.name for w in self.train_websites]
        val_website_names = [w.name for w in self.val_websites]
        print(f'{train_website_names=}, {val_website_names=}')

        # Take subsets.
        subsetter = Subsetter()
        self.train_pages = subsetter(self.train_websites, self.params.train_subset)
        self.val_pages = subsetter(self.val_websites, self.params.val_subset)
        self.test_pages = subsetter(self.val_websites, self.params.test_subset)
        print(f'{len(self.train_pages)=}, {len(self.val_pages)=}, {len(self.test_pages)=}')

    def create_dataloaders(self):
        """
        Samples nodes for train/val/test subsets of pages and extracts features
        for them. Convenience wrapper for `create_dataloader`.
        """

        # Create dataloaders.
        self.train_loader = self.create_dataloader(self.train_pages, 'train',
            shuffle=True,
            train=True,
        )
        self.val_loader = self.create_dataloader(self.val_pages, 'val')
        self.test_loader = self.create_dataloader(self.test_pages, 'test',
            lazy=True
        )

    def create_dataloader(self,
        pages: list[awe.data.set.pages.Page],
        desc: str,
        shuffle: bool = False,
        train: bool = False,
        lazy: bool = False
    ):
        """
        Creates a `torch.utils.data.DataLoader` by sampling nodes from `pages`
        and extracting their features.

        This is mainly for internal use. Externally, it is recommended to use
        `create_run` instead.

        Parameters:
        - `desc`: name for reporting, e.g., `train`, `val`, `test`,
        - `shuffle`: shuffle node samples (should be used only for training),
        - `train`: update feature context (otherwise, it's frozen),
        - `lazy`: do not cache prepared samples.
        """

        set_seed(42)

        flags = {
            'shuffle': shuffle,
            'train': train,
            'lazy': lazy,
        }
        if any(flags.values()):
            desc = f'{desc} ({", ".join(k for k, v in flags.items() if v)})'

        sampler = awe.data.sampling.Sampler(
            trainer=self,
            pages=pages,
            desc=desc,
            train=train
        )
        if lazy:
            nodes = awe.data.sampling.LazySampler(sampler)
        else:
            nodes = sampler.load()
            num_labels = collections.Counter(
                None if not n.label_keys else n.label_keys[0][0]
                for n in nodes
            )
            print(f'Sampled {desc!r} nodes {dict(num_labels)}.', file=sys.stderr)
        return torch.utils.data.DataLoader(
            nodes,
            batch_size=self.params.batch_size,
            collate_fn=awe.data.sampling.Collater(),
            shuffle=shuffle,
        )

    def create_run(self,
        pages: list[awe.data.set.pages.Page],
        desc: str,
        log: bool = False,
        shuffle: bool = False,
        lazy: bool = False,
    ):
        """
        Creates `RunInput`. Convenience wrapper for `create_dataloader`.

        Parameters:
        - `desc`: name for reporting, e.g., `train`, `val`, `test`,
        - `log`: enable TensorBoard logging,
        - `shuffle`: shuffle node samples (should be used only for training),
        - `lazy`: do not cache prepared samples.
        """

        return RunInput(
            pages=pages,
            name=desc,
            loader=self.create_dataloader(
                pages=pages,
                desc=desc,
                shuffle=shuffle,
                lazy=lazy,
            ),
            prefix=desc if log else None,
            lazy=lazy,
        )

    def explore_data(self):
        """
        Simple data exploration.

        Shows target nodes and what the model will see for them (according to
        current hyper-parameters).
        """

        nodes: list[awe.data.graph.dom.Node] = self.train_loader.dataset

        def get_text(node: awe.data.graph.dom.Node):
            if not self.params.tokenize_node_attrs:
                return ''
            attrs = awe.features.text.get_node_attr_text(
                node=node,
                params=self.params
            )
            return awe.features.text.humanize_string(attrs)

        return pd.DataFrame(
            {
                'label_key': node.label_keys[0],
                'value': node.get_text_or_tag(),
                'url': node.dom.page.url,
            } | (
                node.parent.visuals if node.is_text else node.visuals
            ) | {
                'attrs': t for t in [get_text(node)]
                if not self.params.tokenize_node_attrs_only_ancestors and t
            } | {
                f'anc_{i}': f'<{a.html_tag}>[{get_text(a)}]'
                for i, a in (
                    enumerate(
                        node.iterate_ancestors(self.params.n_ancestors)
                        if self.params.n_ancestors is not None
                        else node.iterate_all_ancestors()
                    )
                    if self.params.ancestor_chain
                    else ()
                )
            } | {
                f'neighbor_{i}': v.neighbor.get_text_or_tag()
                for i, v in (
                    enumerate(node.visual_neighbors)
                    if self.params.visual_neighbors
                    else ()
                )
            }
            for node in nodes
            if node.label_keys
        )

    def explore_visuals(self):
        """Displays statistics for prepared visual features."""

        visuals = self.extractor.get_feature(awe.features.visual.Visuals)
        if visuals is None:
            return None
        return visuals.extraction.describe()

    def create_model(self):
        """Creates fresh model (without any trained weights)."""

        set_seed(42)

        self.evaluator = awe.model.eval.Evaluator(self)
        self.model = awe.model.classifier.Model(self).to(self.device)

    def create_version(self):
        """
        Creates version directory and initializes TensorBoard logging in it.

        The previous version directory is automatically removed if it has the
        same name (so it is effectively overwritten by this new version). See
        `Version.delete_last`.

        If hyper-parameter `restore_num` is specified, a previous version is
        restored instead of creating a new version.
        """

        if self.params.restore_num is not None:
            self.version = awe.training.versioning.Version(
                number=self.params.restore_num,
                name=self.params.version_name
            )
            self.restore_version(self.version)
            print(self.restore_model())
        else:
            awe.training.versioning.Version.delete_last(self.params.version_name)
            self.version = awe.training.versioning.Version.create_new(
                self.params.version_name
            )
            self.restored_state = None

            # Save params.
            self.params.save_version(self.version)

            # Save model details.
            s = torchinfo.summary(self.model, verbose=0)
            print(f'Saving {self.version.model_text_path!r}.')
            with open(self.version.model_text_path, mode='w', encoding='utf-8') as f:
                f.write(str(s))

        # Initialize TensorBoard logger.
        self.writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=self.version.version_dir_path,
        )

    def restore_version(self, version: awe.training.versioning.Version):
        """Restores last model checkpoint of the specified `version`."""

        checkpoints = version.get_checkpoints()
        if len(checkpoints) == 0:
            raise RuntimeError(f'No checkpoints ({version.version_dir_path!r})')

        # Check params.
        restored_params = awe.training.params.Params.load_version(version)
        difference = self.params.difference(restored_params,
            ignore_vars=('restore_num', 'epochs')
        )
        if difference:
            raise RuntimeError(
                f'Restored params differ from current params ({difference}).')

        self.restore_checkpoint(checkpoints[-1])

    def restore_checkpoint(self, checkpoint: awe.training.versioning.Checkpoint):
        """Restores the specified model `checkpoint`."""

        print(f'Loading {checkpoint.file_path!r}...')
        self.restored_state = torch.load(checkpoint.file_path,
            # Load GPU tensors to CPU if GPU is not available.
            map_location=self.device if not torch.cuda.is_available() else None
        )

    def restore_features(self):
        """Restores feature context (after calling `restore_checkpoint`)."""

        print('Restoring features...')
        self.label_map = self.restored_state['label_map']
        self.extractor.restore_features(self.restored_state['features'])

    def restore_model(self):
        """Restores model weights (after calling `restore_checkpoint`)."""

        print('Restoring model state...')
        return self.model.load_state_dict(self.restored_state['model'])

    def _reset_loop(self):
        """Resets variables tracking state of a train/val/predict loop."""

        set_seed(42)
        self.step = 0
        self.train_progress = None
        self.val_progress = None

    def _finalize(self):
        """Cleanup of a train/val/predict loop."""

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
        """
        Creates `RunInput`s `train`, `val` and executes training with validation
        after each epoch.

        If `restore_checkpoint` has been called, training state is restored.
        """

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
            name='train',
            prefix='train',
            progress=lambda: self.train_progress,
            progress_metrics=train_progress_metrics
        )
        val_run = RunInput(
            pages=self.val_pages,
            loader=self.val_loader,
            name='val',
            prefix='val',
            progress=lambda: self.val_progress,
            progress_metrics=val_progress_metrics
        )
        remove_last_checkpoint = None
        for epoch_idx in tqdm(range(self.params.epochs), desc='train'):
            if epoch_idx < start_epoch_idx:
                # Skip over restored epochs in this way, so the progress bar is
                # in a proper state.
                continue

            # Test-save a checkpoint to surface errors early (like improperly
            # reloaded Python modules).
            if epoch_idx == 0 and (self.params.save_temporary_checkpoint or
                self.params.save_better_val_loss_checkpoint or
                self.params.save_every_n_epochs
            ):
                self.save_checkpoint(epoch_idx=0).delete()

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

            # Remove temporary checkpoint.
            if remove_last_checkpoint is not None:
                remove_last_checkpoint.delete()
                remove_last_checkpoint = None

            # Save model checkpoint if better loss reached or if Nth epoch.
            if (self.params.save_better_val_loss_checkpoint and
                is_better_val_loss) or (
                self.params.save_every_n_epochs is not None and
                epoch_idx % self.params.save_every_n_epochs == 0
            ):
                self.save_checkpoint(
                    epoch_idx=epoch_idx,
                    best_val_loss=best_val_loss
                )
            elif self.params.save_temporary_checkpoint:
                # Otherwise, save checkpoint temporarily. If training is
                # canceled, it'll be available, otherwise, it'll be deleted in
                # the next epoch. Except for the final epoch, that is always
                # preserved.
                remove_last_checkpoint = self.save_checkpoint(
                    epoch_idx=epoch_idx,
                    best_val_loss=best_val_loss
                )

        self._finalize()

    def save_checkpoint(self, epoch_idx: int, best_val_loss: int = sys.maxsize):
        """Saves model `checkpoint` including training state."""

        ckpt = self.version.create_checkpoint(epoch=epoch_idx, step=self.step)
        state = {
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'step': self.step,
            'epoch': epoch_idx,
            'best_val_loss': best_val_loss,
            'label_map': self.label_map,
            'features': self.extractor.features,
        }
        torch.save(state, ckpt.file_path)

        # Also save JSON for inference UI.
        with open(self.version.info_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_info(), f, indent=2, sort_keys=True)

        return ckpt

    def get_info(self):
        """
        Constructs JSON object conforming to interface `ModelInfo` defined in
        `js/lib/demo/model-info.ts`.
        """

        website_url_domains = [
            os.path.commonprefix([p.url for p in g])
            for _, g in itertools.groupby(
                self.train_pages,
                key=lambda p: p.website.name
            )
        ]
        return {
            'labels': list(self.label_map.label_to_id.keys()),
            'vertical': self.vertical.name,
            'websites': website_url_domains
        }

    def _train_epoch(self, run: RunInput, val_run: RunInput):
        """Trains the model for one epoch with evaluation afterwards."""

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
            if self.params.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.params.gradient_clipping
                )
            self.optim.step()
            self.step += 1
            self.train_progress.update()

            if batch_idx % self.params.log_every_n_steps == 0:
                # Log aggregate train loss.
                self._eval(run, fast_eval)
                fast_eval.clear()

                # Compute all metrics every once in a while.
                slow_eval.add_slow(awe.model.classifier.Prediction(batch, outputs))

            # Validate during training. Always validate in the beginning (step
            # 1) to catch possible bugs in the validation loop.
            if (self.step == 1 or
                (self.params.eval_every_n_steps is not None and
                batch_idx % self.params.eval_every_n_steps == 0)
            ):
                self._validate_epoch(val_run)
                self.model.train()

        # Log aggregate metrics.
        self._eval(run, fast_eval)
        self._eval(run, slow_eval)
        return total_eval.compute()

    def _validate_epoch(self, run: RunInput):
        """Evaluates the model after one epoch of training."""

        with torch.no_grad():
            return self._validate_epoch_core(run)

    def _validate_epoch_core(self, run: RunInput):
        self.model.eval()
        if self.val_progress is None:
            self.val_progress = tqdm(desc=f'val {run.name}')
        self.val_progress.reset(total=len(run.loader))
        evaluation = self.evaluator.start_evaluation()
        for batch in run.loader:
            outputs = self.model.forward(batch)
            evaluation.add(awe.model.classifier.Prediction(batch, outputs))
            if run.lazy:
                self.val_progress.total = len(run.loader)
            self.val_progress.update()
        return self._eval(run, evaluation)

    def _eval(self,
        run: RunInput,
        evaluation: awe.model.eval.Evaluation
    ):
        """Populates progress bar and TensorBoard with evaluation metrics."""

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
        """Evaluates the subset `run` (independently of training)."""

        self._reset_loop()
        metrics = self._validate_epoch(run)
        self._finalize()

        # Save metrics.
        results_file_path = self.version.get_results_path(run.name)
        with open(results_file_path, mode='w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(f'Saved {run.name!r} to {results_file_path!r}.')

        return metrics

    def test(self):
        """
        Like `validate`, but the set of pages for training is created in
        `create_dataloader` according to hyper-parameters.
        """

        test_run = RunInput(
            pages=self.test_pages,
            name='test',
            loader=self.test_loader,
            progress=lambda: self.val_progress,
            progress_metrics=('loss', 'f1/page'),
            lazy=True,
        )
        return self.validate(test_run)

    def predict(self, run: RunInput):
        """Obtains model predictions for the subset `run`."""

        predictions: list[awe.model.classifier.Prediction] = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(run.loader, desc='pred'):
                outputs = self.model.forward(batch)
                predictions.append(awe.model.classifier.Prediction(batch, outputs))
        return predictions

    def decode(self, preds: list[awe.model.classifier.Prediction]):
        """A convenience wrapper for `Decoder.decode`."""

        return awe.model.decoding.Decoder(self).decode(preds)

    def decode_raw(self, preds: list[awe.model.classifier.Prediction]):
        """A convenience wrapper for `Decoder.decode_raw`."""

        return awe.model.decoding.Decoder(self).decode_raw(preds)

def set_seed(seed: int):
    """
    Utility for fixing seeds, so all randomness is reproducible.

    This is called inside each `Trainer`'s top-level method, so repeated
    invocations are always deterministic (e.g., in Jupyter notebooks).
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
