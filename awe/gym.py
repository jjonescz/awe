import dataclasses
import datetime
import os
import re
import shutil
from dataclasses import dataclass
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from tqdm.auto import tqdm

from awe import awe_model, utils
from awe.data import dataset

LOG_DIR = 'logs'

def get_version_path(version: int):
    return f'{LOG_DIR}/version_{version}'

@dataclass
class Checkpoint:
    path: str
    version: int
    epoch: int
    step: int

    @property
    def keys(self):
        return (self.version, self.epoch, self.step)

    @property
    def version_path(self):
        return get_version_path(self.version)

    @property
    def model_path(self):
        return f'{self.version_path}/model.pkl'

    @property
    def model_text_path(self):
        return f'{self.version_path}/model.txt'

    def get_results_path(self, dataset_name: str):
        return f'{self.version_path}/results-{dataset_name}.txt'

    @property
    def inputs_path(self):
        return f'{self.version_path}/inputs.txt'

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
    restore_checkpoint: Optional[Union[str, bool]] = None
    trainer: Optional[pl.Trainer] = None

    def __init__(self,
        ds: dataset.DatasetCollection,
        model: awe_model.AweModel
    ):
        self.ds = ds
        self.model = model

    def get_versions(self):
        for dirname in os.listdir(LOG_DIR):
            match = re.match(r'version_(\d+)', dirname)
            if match is not None:
                yield int(match.group(1))

    def get_checkpoints(self, base_path: str):
        if not os.path.exists(base_path):
            return []
        for filename in os.listdir(base_path):
            match = re.match(r'epoch=(\d+)-step=(\d+)\.ckpt', filename)
            epoch = int(match.group(1))
            step = int(match.group(2))
            yield Checkpoint(f'{base_path}/{filename}', None, epoch, step)

    def get_all_checkpoints(self):
        """Obtains checkpoints across all versions."""
        for version in self.get_versions():
            checkpoints_dir = f'{get_version_path(version)}/checkpoints'
            for ckpt in self.get_checkpoints(checkpoints_dir):
                yield dataclasses.replace(ckpt, version=version)

    def get_last_checkpoint(self):
        """Latest of `get_all_checkpoints`."""
        checkpoints = list(self.get_all_checkpoints())
        if len(checkpoints) == 0:
            return None

        return utils.where_max(checkpoints, lambda c: c.keys)

    def get_last_checkpoint_path(self):
        if self.restore_checkpoint is not None:
            if self.restore_checkpoint is False: # user disabled checkpoint
                return None
            return self.restore_checkpoint

        last_checkpoint = self.get_last_checkpoint()
        return last_checkpoint.path if last_checkpoint is not None else None

    def get_last_checkpoint_version(self):
        if self.restore_checkpoint is False: # user disabled checkpoint
            return None

        if self.restore_checkpoint is not None:
            # Specific checkpoint restored, but unknown version.
            return None

        last_checkpoint = self.get_last_checkpoint()
        return last_checkpoint.version if last_checkpoint is not None else None

    def save_model(self):
        path = self.get_last_checkpoint().model_path
        torch.save(self.model, path)
        return path

    def save_model_text(self):
        path = self.get_last_checkpoint().model_text_path
        model_text = f'{self.model}\n\n{self.model.summarize(max_depth=1)}'
        return utils.save_or_check_file(path, model_text)

    def save_results(self, dataset_name: str):
        results = self.trainer.validate(
            self.model,
            self.ds[dataset_name].loader
        )

        path = self.get_last_checkpoint().get_results_path(dataset_name)
        return utils.save_or_check_file(path, str(results))

    def save_inputs(self):
        """
        Saves inputs (list of pages, batch size) used for training and
        validation.
        """
        path = self.get_last_checkpoint().inputs_path
        text = str(self.ds.extract_inputs())
        utils.save_or_check_file(path, text)
        return text

    def save_named_version(self, name: str):
        """Saves the last version with a name."""
        last_checkpoint = self.get_last_checkpoint()
        version_path = f'{LOG_DIR}/saved_version_{last_checkpoint.version}_{name}'
        if os.path.isdir(version_path):
            raise RuntimeError(f'Directory already exists: {version_path}')
        return shutil.copytree(last_checkpoint.version_path, version_path)

    def create_logger(self):
        return TensorBoardLogger(
            save_dir=os.getcwd(),
            name=LOG_DIR,
            version=self.get_last_checkpoint_version()
        )
