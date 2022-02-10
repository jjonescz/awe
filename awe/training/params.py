import dataclasses
import json
import os
import warnings
from typing import Optional

import awe.data.constants
import awe.training.logging


@dataclasses.dataclass
class Params:
    # General training
    train_subset: int = 2000
    val_subset: int = 50
    epochs: int = 5
    version_name: str = ''
    batch_size: int = 16
    save_every_n_epochs: Optional[int] = 1
    log_every_n_steps: int = 10
    eval_every_n_steps: Optional[int] = 50
    use_gpu: bool = True

    @classmethod
    def load_version(cls, version: awe.training.logging.Version):
        return cls.load_file(version.params_path)

    @classmethod
    def load_user(cls):
        """Loads params from user-provided file."""
        path = f'{awe.data.constants.DATA_DIR}/params.json'
        if not os.path.exists(path):
            # Create file with default params as template.
            warnings.warn(f'No params file, creating one at {repr(path)}.')
            Params().save_file(path)
            return None
        return cls.load_file(path)

    @staticmethod
    def load_file(path: str):
        with open(path, mode='r', encoding='utf-8') as f:
            return Params(**json.load(f))

    def save_version(self, version: awe.training.logging.Version):
        self.save_file(version.params_path)

    def save_file(self, path: str):
        with open(path, mode='w', encoding='utf-8') as f:
            json.dump(dataclasses.asdict(self), f,
                indent=2,
                sort_keys=True
            )

    def update_from(self, checkpoint: awe.training.logging.Checkpoint):
        self.epochs = checkpoint.epoch + 1
