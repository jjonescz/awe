import dataclasses
import os
import re
from dataclasses import dataclass
from typing import Optional, Union

import torch

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

class Gym:
    checkpoint: Optional[Union[str, bool]] = None

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
