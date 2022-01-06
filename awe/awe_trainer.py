import dataclasses
from typing import Sequence

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
    label_weights: Sequence[int] = (1,) + (300,) * 4

    # Training
    epochs: int = 10

def train(params: AweTrainingParams):
    # Split data.
    sds = swde.Dataset(suffix='-exact')
    websites = sds.verticals[0].websites
    rng = np.random.default_rng(42)
    train_pages = [
        p for i in params.training_website_indices
        for p in websites[i].pages
    ]
    val_pages = [
        p for i in range(len(websites))
        if i not in params.training_website_indices
        for p in rng.choice(websites[i].pages, 50, replace=False)
    ]

    # Load data.
    ds = dataset.DatasetCollection()
    ds.root.freeze()
    ds.features = [
        features.Depth(),
        features.IsLeaf(),
        features.CharCategories(),
        features.Visuals(),
        features.CharIdentifiers(),
        features.WordIdentifiers()
    ]
    ds.create('train', train_pages, shuffle=True)
    ds.create('val_unseen', val_pages)
    ds.create('val_seen', rng.choice(train_pages, 200, replace=False))
    ds.create_dataloaders(
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )

    # Create model.
    label_count = len(ds.first_dataset.label_map)
    model = awe_model.AweModel(
        feature_count=ds.feature_dim,
        label_count=label_count,
        label_weights=params.label_weights,
        char_count=len(ds.root.chars) + 1,
        use_gnn=True,
        use_lstm=True,
        use_cnn=False,
        lstm_args={ 'bidirectional': True },
        filter_node_words=True,
        label_smoothing=0.1
    )

    # Train model.
    g = gym.Gym(ds, model)
    g.restore_checkpoint = False
    g.trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=params.epochs,
        callbacks=[gym.CustomProgressBar(refresh_rate=10)],
        resume_from_checkpoint=g.get_last_checkpoint_path(),
        logger=g.create_logger(),
    )
    g.trainer.fit(model, data_module.DataModule(ds))

    # Save results.
    g.save_results('val_unseen')
    g.save_results('val_seen')
