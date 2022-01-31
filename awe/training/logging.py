import dataclasses
import os
import re
import shutil

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import awe.utils

LOG_DIR = 'logs'

@dataclasses.dataclass
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
        return awe.utils.where_max(existing_versions, lambda v: v.number)

    @classmethod
    def delete_last(cls, name: str):
        """
        If last version has the given `name` or no name, deletes it.
        """

        latest_version = cls.get_latest()
        if latest_version is not None and (
            latest_version.name in (name, '')
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
    def bak_dir_path(self):
        # We use separate subfolder so that the trashed logdirs don't get loaded
        # alongside normal versions.
        return f'{LOG_DIR}/bak/{self.version_dir_name}'

    @property
    def checkpoints_dir_path(self):
        return f'{self.version_dir_path}/checkpoints'

    def get_checkpoints(self):
        if not os.path.exists(self.checkpoints_dir_path):
            return
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

    @property
    def params_path(self):
        return f'{self.version_dir_path}/params.json'

    def exists(self):
        return os.path.exists(self.version_dir_path)

    def create(self):
        os.makedirs(self.version_dir_path, exist_ok=True)

    def delete(self):
        if not os.path.exists(self.version_dir_path):
            return

        if os.path.exists(self.bak_dir_path):
            print(f'Deleting {self.bak_dir_path}')
            shutil.rmtree(self.bak_dir_path)

        print(f'Trashing {self.version_dir_path}')
        os.rename(self.version_dir_path, self.bak_dir_path)

    def create_logger(self):
        return TensorBoardLogger(
            save_dir=os.getcwd(),
            name=LOG_DIR,
            version=self.version_dir_name
        )

@dataclasses.dataclass
class Checkpoint:
    version: Version
    epoch: int
    step: int

    @property
    def file_name(self):
        return f'epoch={self.epoch}-step={self.step}.ckpt'

    @property
    def file_path(self):
        return f'{self.version.checkpoints_dir_path}/{self.file_name}'