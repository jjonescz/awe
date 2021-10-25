import dataclasses
import os
import re
from dataclasses import dataclass
from typing import Optional, Union
import shutil

import torch
import pytorch_lightning as pl

from awe import awe_model, utils
from awe.data import dataset

LOG_DIR = 'lightning_logs'

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

    def get_results_path(self, dataset_name):
        return f'{self.version_path}/results-{dataset_name}.txt'

class Gym:
    checkpoint: Optional[Union[str, bool]] = None
    trainer: Optional[pl.Trainer] = None

    def __init__(self,
        ds: dataset.Dataset,
        model: awe_model.AweModel
    ):
        self.ds = ds
        self.model = model

    def get_versions(self):
        for dirname in os.listdir(LOG_DIR):
            yield int(re.match(r'version_(\d+)', dirname).group(1))

    def get_checkpoints(self, base_path: str):
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
        if self.checkpoint is not None:
            if self.checkpoint is False: # user-disabled checkpoint
                return None
            return self.checkpoint

        last_checkpoint = self.get_last_checkpoint()
        return last_checkpoint.path if last_checkpoint is not None else None

    def save_model(self):
        path = self.get_last_checkpoint().model_path
        torch.save(self.model, path)
        return path

    def save_model_text(self):
        path = self.get_last_checkpoint().model_text_path
        with open(path, mode='w', encoding='utf-8') as f:
            f.write(str(self.model))
        return path

    def save_results(self, dataset_name: str):
        results = self.trainer.validate(
            self.model,
            self.ds.loaders[dataset_name]
        )

        path = self.get_last_checkpoint().get_results_path(dataset_name)
        with open(path, mode='w', encoding='utf-8') as f:
            f.write(str(results))
        return path

    def save_named_version(self, name: str):
        """Saves the last version with a name."""
        version_path = f'{LOG_DIR}/{name}'
        if os.path.isdir(version_path):
            raise RuntimeError(f'Directory already exists: {version_path}')
        return shutil.copytree(
            self.get_last_checkpoint().version_path,
            version_path
        )
