import os
import re
from typing import Optional, Union

from awe import awe_model, utils
from awe.data import dataset

LOG_DIR = 'lightning_logs'

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
            yield f'{base_path}/{filename}', epoch, step

    def get_all_checkpoints(self):
        """Obtains checkpoints across all versions."""
        for version in self.get_versions():
            checkpoints_dir = f'{LOG_DIR}/version_{version}/checkpoints'
            for path, epoch, step in self.get_checkpoints(checkpoints_dir):
                yield path, version, epoch, step

    def get_checkpoint(self):
        if self.checkpoint is not None:
            if self.checkpoint is False: # user-disabled checkpoint
                return None
            return self.checkpoint

        checkpoints = list(self.get_all_checkpoints())
        if len(checkpoints) == 0:
            return None

        last_checkpoint = utils.where_max(checkpoints, lambda t: t[1:])
        return last_checkpoint[0]
