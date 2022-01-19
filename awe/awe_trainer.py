import sys
from dataclasses import dataclass
from typing import Optional, Sequence, TypeVar

import numpy as np
import pytorch_lightning as pl
import torch

from awe import awe_model, features, gym
from awe.data import data_module, dataset, swde


@dataclass
class AweTrainingParams:
    # Dataset split
    training_website_indices: Sequence[int] = (0, 3, 4, 5, 7)
    """Only `auto` vertical for now."""

    train_pages_subset: Optional[int] = None
    val_pages_subset: Optional[int] = 50
    seen_pages_subset: Optional[int] = 200

    # Data loading
    batch_size: int = 64
    num_workers: int = 8

    # Feature extraction
    feat: features.AweFeatureOptions = features.AweFeatureOptions()

    # Model
    model: awe_model.AweModelParams = awe_model.AweModelParams()

    # Training
    use_gpu: bool = True
    epochs: int = 10

    version_name: Optional[str] = None
    """Suffix of version folder where logs are saved."""

    delete_existing_version: bool = False
    """If version with the same name already exists, it'll be deleted."""

T = TypeVar('T')

class DataSplitterFactory:
    def __init__(self):
        self.rng = np.random.default_rng(42)

    def create(self):
        return DataSplitter(seed=self.rng.integers(0, sys.maxsize))

class DataSplitter:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def get_subset(self, data: list[T], amount: Optional[int]) -> list[T]:
        if amount is None:
            return data
        return self.rng.choice(data, amount, replace=False)

class AweTrainer:
    ds: dataset.DatasetCollection
    datamodule: data_module.DataModule
    g: gym.Gym
    interrupted: bool = False

    def __init__(self, params: AweTrainingParams):
        self.params = params

    def load_data(self):
        pl.seed_everything(42, workers=True)

        # Delete existing version.
        if self.params.delete_existing_version:
            gym.Version.delete_last(self.params.version_name)

        # Split data.
        sds = swde.Dataset(suffix='-exact')
        websites = sds.verticals[0].websites
        rand = DataSplitterFactory()
        train_rand = rand.create()
        val_rand = rand.create()
        seen_rand = rand.create()
        train_pages = [
            p for i in self.params.training_website_indices
            for p in train_rand.get_subset(
                websites[i].pages, self.params.train_pages_subset)
        ]
        val_pages = [
            p for i in range(len(websites))
            if i not in self.params.training_website_indices
            for p in val_rand.get_subset(
                websites[i].pages, self.params.val_pages_subset)
        ]

        # Load data.
        self.ds = dataset.DatasetCollection()
        self.ds.root.freeze()
        self.ds.features = [
            features.Depth(),
            features.IsLeaf(),
            features.CharCategories(),
            features.Visuals(),
            features.CharIdentifiers(),
            features.WordIdentifiers()
        ]
        self.ds.create('train', train_pages, shuffle=True)
        self.ds.create('val_unseen', val_pages)
        self.ds.create('val_seen',
            seen_rand.get_subset(train_pages, self.params.seen_pages_subset))
        self.ds.create_dataloaders(
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )
        self.datamodule = data_module.DataModule(self.ds)
        return self.ds.get_lengths()

    def extract_features(self):
        # Prepare feature context (e.g., word dictionaries).
        prev_root = self.ds.root.describe()
        print(f'Before: {prev_root}')
        self.ds.prepare_features(parallelize=8)
        self.ds.root.freeze()
        curr_root = self.ds.root.describe()
        print(f'After: {curr_root}')
        saved = self.ds.save_root_context(
            overwrite_existing=(prev_root['pages'] != curr_root['pages'])
        )
        print(f'Saved: {saved}')

        # Compute features.
        self.ds.compute_features(parallelize=6)

    def prepare_model(self):
        # Create model.
        label_count = len(self.ds.first_dataset.label_map)
        model = awe_model.AweModel(
            feature_count=self.ds.feature_dim,
            label_count=label_count,
            char_count=len(self.ds.root.chars) + 1,
            params=self.params.model,
        )

        # Prepare model for training.
        self.g = gym.Gym(self.ds, model, self.params.version_name)
        self.g.trainer = pl.Trainer(
            gpus=torch.cuda.device_count() if self.params.use_gpu else 0,
            max_epochs=self.params.epochs,
            callbacks=[
                gym.CustomProgressBar(refresh_rate=10),
                InterruptDetectingCallback(self),
            ],
            logger=self.g.create_logger(),
            gradient_clip_val=0.5,
        )

    def find_lr(self):
        lr_finder = self.g.trainer.tuner.lr_find(self.g.model, self.datamodule)
        print(lr_finder.results)
        lr_finder.plot(suggest=True, show=True)

    def train_model(self):
        # Save training inputs.
        self.g.save_inputs()
        self.g.save_model_text()

        # Train model.
        self.g.trainer.fit(self.g.model, self.datamodule)

        # Save results.
        self.g.save_results('val_unseen')
        self.g.save_results('val_seen')

def train_grid(param_grid: list[AweTrainingParams]):
    for params in param_grid:
        print(f'*** Version: {params.version_name} ***')

        trainer = AweTrainer(params)
        lengths = trainer.load_data()
        print(lengths)

        trainer.prepare_model()
        trainer.train_model()

        # End early if interrupted.
        if trainer.interrupted:
            print(f'Training of {params.version_name} interrupted')
            break

        status = trainer.g.trainer.state.status
        print(f'Training of {params.version_name} {status}')

class InterruptDetectingCallback(pl.Callback):
    def __init__(self, trainer: AweTrainer):
        self.trainer = trainer

    def on_keyboard_interrupt(self, *_):
        self.trainer.interrupted = True
