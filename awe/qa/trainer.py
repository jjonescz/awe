import dataclasses
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data

from awe import awe_graph, gym, qa_model
from awe.data import qa_dataset, swde

#  Ignore warnings.
warnings.filterwarnings('ignore', message='__floordiv__ is deprecated')

@dataclasses.dataclass
class QaTrainerParams:
    train_subset: int = 2000
    val_subset: int = 50
    epochs: int = 5
    version_name: str = ''
    batch_size: int = 16

class QaTrainer:
    train_pages: list[awe_graph.HtmlPage]
    val_pages: list[awe_graph.HtmlPage]
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    model: qa_model.QaModel
    g: gym.Gym
    trainer: pl.Trainer

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
        self.train_pages = rng.choice(train_pages, self.params.train_subset, replace=False)
        self.val_pages = rng.choice(val_pages, self.params.val_subset, replace=False)
        print(f'{len(train_pages)=}, {len(val_pages)=}')

        # Create dataloaders.
        self.train_loader = self.create_dataloader(self.train_pages, shuffle=True)
        self.val_loader = self.create_dataloader(self.val_pages)

    def create_dataloader(self,
        pages: list[awe_graph.HtmlPage],
        shuffle: bool = False
    ):
        return torch.utils.data.DataLoader(
            qa_dataset.QaEntryLoader(pages),
            batch_size=self.params.batch_size,
            collate_fn=qa_dataset.QaCollater(self.pipeline.tokenizer),
            shuffle=shuffle,
        )

    def prepare_data(self):
        qa_dataset.prepare_entries(self.train_pages)
        qa_dataset.prepare_entries(self.val_pages)

    def create_model(self):
        self.model = qa_model.QaModel(self.pipeline)

    def delete_previous_version(self):
        gym.Version.delete_last(self.params.version_name)

    def train(self):
        self.g = gym.Gym(None, None, version_name=self.params.version_name)
        self.trainer = pl.Trainer(
            gpus=torch.cuda.device_count(),
            max_epochs=self.params.epochs,
            logger=self.g.create_logger(),
        )
        self.trainer.fit(self.model, self.train_loader, self.val_loader)

    def restore(self, checkpoint_name: str):
        # Load model from checkpoint.
        self.g = gym.Gym(None, None, version_name='')
        self.trainer = pl.Trainer(
            gpus=torch.cuda.device_count(),
            max_epochs=self.params.epochs,
            logger=self.g.create_logger(),
            resume_from_checkpoint=checkpoint_name
        )
        self.trainer.fit(self.model, self.train_loader)

    def validate(self):
        self.trainer.validate(self.model, self.val_loader)

    def validate_seen(self):
        loader = self.create_dataloader(self.train_pages[:2])
        self.trainer.validate(self.model, loader)

    def predict_examples(self):
        loader = self.create_dataloader(self.val_pages[:2])
        preds = self.trainer.predict(self.model, loader)
        decoder = qa_dataset.QaDecoder(self.pipeline.tokenizer)
        return decoder.decode_predictions(preds)

    def predict_seen_examples(self):
        loader = self.create_dataloader(self.train_pages[:2])
        preds = self.trainer.predict(self.model, loader)
        decoder = qa_dataset.QaDecoder(self.pipeline.tokenizer)
        return decoder.decode_predictions(preds)
