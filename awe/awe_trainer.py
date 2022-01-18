import dataclasses
from typing import Any, Sequence

import numpy as np
import pytorch_lightning as pl
import torch

from awe import awe_model, features, gym
from awe.data import data_module, dataset, swde


@dataclasses.dataclass
class AweTrainingParams:
    # Dataset split
    training_website_indices: Sequence[int] = (0, 3, 4, 5, 7)
    """Only `auto` vertical for now."""

    # Data loading
    batch_size: int = 64
    num_workers: int = 8

    # Model
    use_lstm: bool = True
    label_weights: Sequence[int] = (1,) + (300,) * 4
    lstm_args: dict[str, Any] = None

    # Training
    epochs: int = 10

class AweTrainer:
    ds: dataset.DatasetCollection

    def __init__(self, params: AweTrainingParams):
        self.params = params

    def load_data(self):
        pl.seed_everything(42, workers=True)

        # Split data.
        sds = swde.Dataset(suffix='-exact')
        websites = sds.verticals[0].websites
        rng = np.random.default_rng(42)
        train_pages = [
            p for i in self.params.training_website_indices
            for p in websites[i].pages
        ]
        val_pages = [
            p for i in range(len(websites))
            if i not in self.params.training_website_indices
            for p in rng.choice(websites[i].pages, 50, replace=False)
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
        self.ds.create('val_seen', rng.choice(train_pages, 200, replace=False))
        self.ds.create_dataloaders(
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )
        return self.ds.get_lengths()

    def train_model(self):
        # Create model.
        label_count = len(self.ds.first_dataset.label_map)
        model = awe_model.AweModel(
            feature_count=self.ds.feature_dim,
            label_count=label_count,
            label_weights=self.params.label_weights,
            char_count=len(self.ds.root.chars) + 1,
            use_gnn=True,
            use_lstm=self.params.use_lstm,
            use_cnn=False,
            lstm_args=self.params.lstm_args,
            filter_node_words=True,
            label_smoothing=0.1
        )

        # Prepare model for training.
        g = gym.Gym(self.ds, model)
        g.restore_checkpoint = False
        g.trainer = pl.Trainer(
            gpus=torch.cuda.device_count(),
            max_epochs=self.params.epochs,
            callbacks=[gym.CustomProgressBar(refresh_rate=10)],
            resume_from_checkpoint=g.get_last_checkpoint_path(),
            logger=g.create_logger(),
            gradient_clip_val=0.5,
        )

        # Save training inputs.
        g.save_inputs()
        g.save_model_text()

        # Train model.
        g.trainer.fit(model, data_module.DataModule(self.ds))

        # Save results.
        g.save_results('val_unseen')
        g.save_results('val_seen')
