import dataclasses
import json
import warnings
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import awe.qa.collater
import awe.qa.decoder
import awe.qa.model
import awe.qa.pipeline
import awe.qa.sampler
import awe.training.logging
from awe import awe_graph
from awe.data import swde

#  Ignore warnings.
warnings.filterwarnings('ignore', message='__floordiv__ is deprecated')

@dataclasses.dataclass
class TrainerParams:
    train_subset: int = 2000
    val_subset: int = 50
    epochs: int = 5
    version_name: str = ''
    batch_size: int = 16

    @staticmethod
    def load(version: awe.training.logging.Version):
        with open(version.params_path, mode='r', encoding='utf-8') as f:
            return TrainerParams(**json.load(f))

    def save(self, version: awe.training.logging.Version):
        with open(version.params_path, mode='w', encoding='utf-8') as f:
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

    def __init__(self, params: TrainerParams):
        self.params = params
        self.pipeline = awe.qa.pipeline.Pipeline()

    def create_version(self):
        awe.training.logging.Version.delete_last(self.params.version_name)
        self.version = awe.training.logging.Version.create_new(
            self.params.version_name
        )

        # Save params.
        self.params.save(self.version)

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
            collate_fn=awe.qa.collater.Collater(self.pipeline.tokenizer),
            shuffle=shuffle,
        )

    def create_model(self):
        self.model = awe.qa.model.Model(self.pipeline.model)

    def train(self):
        self._create_trainer(
            logger=self.version.create_logger()
        )
        self.trainer.fit(self.model, self.train_loader, self.val_loader)

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
            resume_from_checkpoint=resume_from_checkpoint
        )

    def validate(self):
        self.trainer.validate(self.model, self.val_loader)

    def validate_seen(self):
        loader = self.create_dataloader(self.train_pages[:2])
        self.trainer.validate(self.model, loader)

    def predict_examples(self):
        loader = self.create_dataloader(self.val_pages[:2])
        preds = self.trainer.predict(self.model, loader)
        decoder = awe.qa.decoder.Decoder(self.pipeline.tokenizer)
        return decoder.decode_predictions(preds)

    def predict_seen_examples(self):
        loader = self.create_dataloader(self.train_pages[:2])
        preds = self.trainer.predict(self.model, loader)
        decoder = awe.qa.decoder.Decoder(self.pipeline.tokenizer)
        return decoder.decode_predictions(preds)
