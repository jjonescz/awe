import dataclasses
from typing import Any, Optional, Sequence

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
    disable_direct_features: bool = False
    use_two_gcn_layers: bool = True
    use_word_vectors: bool = True

    # Training
    epochs: int = 10

    version_name: Optional[str] = None
    """Suffix of version folder where logs are saved."""

class AweTrainer:
    ds: dataset.DatasetCollection
    g: gym.Gym
    interrupted: bool = False

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
            label_smoothing=0.1,
            use_two_gcn_layers=self.params.use_two_gcn_layers,
            disable_direct_features=self.params.disable_direct_features,
            use_word_vectors=self.params.use_word_vectors,
        )

        # Prepare model for training.
        self.g = gym.Gym(self.ds, model, self.params.version_name)
        self.g.trainer = pl.Trainer(
            gpus=torch.cuda.device_count(),
            max_epochs=self.params.epochs,
            callbacks=[
                gym.CustomProgressBar(refresh_rate=10),
                InterruptDetectingCallback(self),
            ],
            logger=self.g.create_logger(),
            gradient_clip_val=0.5,
        )

        # Save training inputs.
        self.g.save_inputs()
        self.g.save_model_text()

        # Train model.
        self.g.trainer.fit(model, data_module.DataModule(self.ds))

        # Save results.
        self.g.save_results('val_unseen')
        self.g.save_results('val_seen')

def train_grid(param_grid: list[AweTrainingParams]):
    for params in param_grid:
        print(f'*** Version: {params.version_name} ***')

        trainer = AweTrainer(params)
        lengths = trainer.load_data()
        print(lengths)

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
