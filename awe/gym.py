import os
import re
from typing import Optional, Union

from awe import awe_model
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
            yield epoch, step

    def get_checkpoint(self):
        if self.checkpoint is not None:
            if self.checkpoint == False: # user-disabled checkpoint
                return None
            return self.checkpoint

        versions = list(self.get_versions())
        if len(versions) == 0:
            return None

        version = max(versions)
        checkpoints_dir = f'{LOG_DIR}/version_{version}/checkpoints'
        checkpoints = list(self.get_checkpoints(checkpoints_dir))
        if len(checkpoints) == 0:
            return None

        epoch = max(map(lambda t: t[0], checkpoints))
        step = max(map(lambda t: t[1], checkpoints))
        return f'{checkpoints_dir}/epoch={epoch}-step={step}.ckpt'
