import os
import re
from dataclasses import dataclass
import shutil
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from tqdm.auto import tqdm

from awe import awe_model, log_utils, utils
from awe.data import dataset

LOG_DIR = 'logs'

@dataclass
class Version:
    number: int
    name: str

    @staticmethod
    def _iterate_all():
        for dirname in os.listdir(LOG_DIR):
            match = re.match(r'(\d+)-(.+)', dirname)
            if match is not None:
                number = int(match.group(1))
                name = match.group(2)
                yield Version(number, name)

    @classmethod
    def get_all(cls):
        return list(cls._iterate_all())

    @classmethod
    def get_latest(cls):
        existing_versions = cls.get_all()
        if len(existing_versions) == 0:
            return None
        return utils.where_max(existing_versions, lambda v: v.number)

    @classmethod
    def delete_last(cls, name: str):
        """
        If last version has the given `name` or no name, deletes it.
        """

        latest_version = cls.get_latest()
        if latest_version is not None and (
            latest_version.name == name or latest_version.name == ''
        ):
            latest_version.delete()

    @classmethod
    def create_new(cls, name: str):
        """
        Creates new version with the given `name` and auto-incremented number.
        """

        latest_version = cls.get_latest()
        if latest_version is None:
            version = Version(1, name)
        else:
            if latest_version.name == name:
                raise RuntimeError(
                    f'Last version {latest_version.version_dir_name} has ' + \
                    'unchanged name.')

            version = Version(latest_version.number + 1, name)
        version.create()
        return version

    @property
    def version_dir_name(self):
        return f'{self.number}-{self.name}'

    @property
    def version_dir_path(self):
        return f'{LOG_DIR}/{self.version_dir_name}'

    @property
    def checkpoints_dir_path(self):
        return f'{self.version_dir_path}/checkpoints'

    def get_checkpoints(self):
        if not os.path.exists(self.checkpoints_dir_path):
            return []
        for filename in os.listdir(self.checkpoints_dir_path):
            match = re.match(r'epoch=(\d+)-step=(\d+)\.ckpt', filename)
            epoch = int(match.group(1))
            step = int(match.group(2))
            yield Checkpoint(self, epoch, step)

    @property
    def model_path(self):
        return f'{self.version_dir_path}/model.pkl'

    @property
    def model_text_path(self):
        return f'{self.version_dir_path}/model.txt'

    def get_results_path(self, dataset_name: str):
        return f'{self.version_dir_path}/results-{dataset_name}.txt'

    @property
    def inputs_path(self):
        return f'{self.version_dir_path}/inputs.txt'

    def exists(self):
        return os.path.exists(self.version_dir_path)

    def create(self):
        os.makedirs(self.version_dir_path, exist_ok=True)

    def delete(self):
        print(f'Deleting {self.version_dir_path}')
        shutil.rmtree(self.version_dir_path)

@dataclass
class Checkpoint:
    version: Version
    epoch: int
    step: int

# pylint: disable=attribute-defined-outside-init, arguments-differ
class CustomProgressBar(ProgressBar):
    """Disables validation progress bar and shows progress over all epochs."""

    def on_train_start(self, trainer, pl_module):
        self.epoch_progress_bar = tqdm(
            total=trainer.max_epochs,
            dynamic_ncols=True,
            desc='Training'
        )
        return super().on_train_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.epoch_progress_bar.update()
        return super().on_train_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self.epoch_progress_bar.close()
        return super().on_train_end(trainer, pl_module)

    def init_validation_tqdm(self):
        return tqdm(disable=True)

class Gym:
    """Manages versions of log directories."""

    restore_checkpoint: Optional[Union[str, bool]] = None
    trainer: Optional[pl.Trainer] = None

    def __init__(self,
        ds: dataset.DatasetCollection,
        model: awe_model.AweModel,
        version_name: str
    ):
        self.ds = ds
        self.model = model
        self.version = Version.create_new(version_name)

    def save_model(self):
        path = self.version.model_path
        torch.save(self.model, path)
        return path

    def save_model_text(self):
        # Get model summary without logging it to console.
        with log_utils.all_logging_disabled():
            summary = self.model.summarize(max_depth=1)

        path = self.version.model_text_path
        model_text = f'{self.model}\n\n{summary}'
        return utils.save_or_check_file(path, model_text)

    def save_results(self, dataset_name: str):
        try:
            # Disable logging to TensorBoard.
            log_dict = self.model.log_dict
            self.model.log_dict = lambda *args, **kwargs: log_dict(*args, logger=False, **kwargs)

            results = self.trainer.validate(
                self.model,
                self.ds[dataset_name].loader
            )
        finally:
            # Restore logging.
            self.model.log_dict = log_dict

        path = self.version.get_results_path(dataset_name)
        return utils.save_or_check_file(path, str(results))

    def save_inputs(self):
        """
        Saves inputs (list of pages, batch size) used for training and
        validation.
        """
        path = self.version.inputs_path
        text = str(self.ds.extract_inputs())
        utils.save_or_check_file(path, text)
        return text

    def create_logger(self):
        return TensorBoardLogger(
            save_dir=os.getcwd(),
            name=LOG_DIR,
            version=self.version.version_dir_name
        )
