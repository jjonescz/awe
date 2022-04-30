"""
Training versions management (logs, checkpoints).

To change the base logdir, use environment variable `LOGDIR`.
"""

import dataclasses
import os
import re
import shutil
import warnings

import awe.utils

LOG_DIR = os.environ.get('LOGDIR', 'logs')
"""
Directory where all version subdirectories are stored, a.k.a. logdir.

By default `logs`. Can be changed via environment variable `LOGDIR`.
"""

@dataclasses.dataclass
class Version:
    """
    Represents one model version, i.e., one experiment with one set of
    hyper-parameters.
    """

    number: int
    name: str

    @classmethod
    def _iterate_all(cls):
        if not os.path.isdir(LOG_DIR):
            return
        for dirname in os.listdir(LOG_DIR):
            if (version := cls.try_parse(dirname)) is not None:
                yield version

    @staticmethod
    def try_parse(dirname: str):
        """Inverse of `version_dir_name`."""

        match = re.match(r'(\d+)-(.*)', dirname)
        if match is not None:
            number = int(match.group(1))
            name = match.group(2)
            return Version(number, name)
        return None

    @classmethod
    def get_all(cls):
        """List of all versions inside logdir."""
        return list(cls._iterate_all())

    @classmethod
    def get_latest(cls):
        """Finds version with the highest number."""

        existing_versions = cls.get_all()
        if len(existing_versions) == 0:
            return None
        return awe.utils.where_max(existing_versions, lambda v: v.number)

    @classmethod
    def find_by_number(cls, number: int):
        """Finds version with the specified `number`."""

        for v in cls._iterate_all():
            if v.number == number:
                return v
        return None

    @classmethod
    def delete_last(cls, name: str):
        """
        If the last version has the given `name`, deletes it (see `delete`). If
        it has no name, keeps it (will be overwritten).
        """

        latest_version = cls.get_latest()
        if latest_version is not None and latest_version.name != '' \
            and latest_version.name == name:
            latest_version.delete()

    @classmethod
    def create_new(cls, name: str):
        """
        Creates new version with the given `name` and auto-incremented number.
        """

        latest_version = cls.get_latest()
        if latest_version is None:
            version = Version(1, name)
        elif latest_version.name == '' and name == '':
            # Overwrite empty version.
            warnings.warn('Overwriting empty version ' +
                f'({latest_version.version_dir_name!r}).')
            version = Version(latest_version.number, name)
        else:
            if latest_version.name == name:
                raise RuntimeError(
                    f'Last version {latest_version.version_dir_name!r} has ' +
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
        """A directory where checkpoints of this version are stored."""

        return f'{self.version_dir_path}/checkpoints'

    def get_checkpoints(self):
        return sorted(self._iter_checkpoints(), key=lambda c: c.file_path)

    def _iter_checkpoints(self):
        if not os.path.exists(self.checkpoints_dir_path):
            return
        for filename in os.listdir(self.checkpoints_dir_path):
            match = re.match(r'epoch=(\d+)-step=(\d+)\.ckpt', filename)
            epoch = int(match.group(1))
            step = int(match.group(2))
            yield Checkpoint(self, epoch, step)

    def create_checkpoint(self, epoch: int, step: int):
        # Create directory for checkpoints.
        os.makedirs(self.checkpoints_dir_path, exist_ok=True)
        return Checkpoint(self, epoch=epoch, step=step)

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

    @property
    def info_path(self):
        return f'{self.version_dir_path}/info.json'

    def exists(self):
        return os.path.exists(self.version_dir_path)

    def create(self):
        print(f'Creating {self.version_dir_path!r}.')
        os.makedirs(self.version_dir_path, exist_ok=True)

    def delete(self):
        """
        Deletes version directory, first only renaming it (so if it was an
        accident, it can be reverted).
        """

        if not os.path.exists(self.version_dir_path):
            return

        if os.path.exists(self.bak_dir_path):
            print(f'Deleting {self.bak_dir_path!r}.')
            shutil.rmtree(self.bak_dir_path)

        print(f'Trashing {self.version_dir_path!r} -> {self.bak_dir_path!r}.')
        os.makedirs(self.bak_dir_path, exist_ok=True)
        os.rename(self.version_dir_path, self.bak_dir_path)

@dataclasses.dataclass
class Checkpoint:
    """Represents one checkpoint of a model, i.e., all its trained weights."""

    version: Version
    epoch: int
    step: int

    @property
    def file_name(self):
        return f'epoch={self.epoch}-step={self.step}.ckpt'

    @property
    def file_path(self):
        return f'{self.version.checkpoints_dir_path}/{self.file_name}'

    def delete(self):
        os.remove(self.file_path)
