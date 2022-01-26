import dataclasses

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data

from awe import gym, qa_model
from awe.data import qa_dataset, swde


@dataclasses.dataclass
class QaTrainerParams:
    train_subset: int = 2000
    val_subset: int = 50
    epochs: int = 5

class QaTrainer:
    train_ds: qa_dataset.QaTorchDataset
    val_ds: qa_dataset.QaTorchDataset
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    model: qa_model.QaModel

    def __init__(self, params: QaTrainerParams):
        self.params = params
        self.pipeline = qa_model.QaPipeline()

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
        train_pages = rng.choice(train_pages, self.params.train_subset, replace=False)
        val_pages = rng.choice(val_pages, self.params.val_subset, replace=False)
        print(f'{len(train_pages)=}, {len(val_pages)=}')

        # Create dataloaders.
        self.train_ds = qa_dataset.QaTorchDataset(train_pages, self.pipeline.tokenizer)
        self.val_ds = qa_dataset.QaTorchDataset(val_pages, self.pipeline.tokenizer)
        self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=1, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_ds, batch_size=1)

    def prepare_data(self):
        self.train_ds.loader.prepare_entries()
        self.val_ds.loader.prepare_entries()

    def create_model(self):
        self.model = qa_model.QaModel(self.pipeline)

    def train(self):
        g = gym.Gym(None, None, version_name='')
        trainer = pl.Trainer(
            gpus=torch.cuda.device_count(),
            max_epochs=self.params.epochs,
            logger=g.create_logger(),
        )
        trainer.fit(self.model, self.train_loader, self.val_loader)
