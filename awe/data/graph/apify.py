import dataclasses
import os
import warnings

import pandas as pd

import awe.data.constants
import awe.data.graph.pages

DIR = f'{awe.data.constants.DATA_DIR}/apify'

@dataclasses.dataclass
class Dataset(awe.data.graph.pages.Dataset):
    verticals: list['Vertical'] = dataclasses.field(repr=False)

    def __init__(self):
        super().__init__(
            name='apify',
            dir_path=DIR,
        )
        self.verticals = [
            Vertical(dataset=self, name='products')
        ]

@dataclasses.dataclass
class Vertical(awe.data.graph.pages.Vertical):
    dataset: Dataset
    websites: list['Website'] = dataclasses.field(repr=False, default_factory=list)

    def __post_init__(self):
        self.websites = list(self._iterate_websites())

    @property
    def dir_path(self):
        return self.dataset.dir_path

    def _iterate_websites(self):
        if not os.path.exists(self.dir_path):
            warnings.warn(
                f'Dataset directory does not exist ({self.dir_path}).')
            return

        for subdir in sorted(os.listdir(self.dir_path)):
            yield Website(self, subdir)

@dataclasses.dataclass
class Website(awe.data.graph.pages.Website):
    vertical: Vertical

    def __init__(self, vertical: Vertical, dir_name: str):
        super().__init__(
            vertical=vertical,
            name=dir_name,
        )

        # Convert dataset.
        if not os.path.exists(self.dataset_pickle_path):
            warnings.warn('Saving dataset in efficient binary format ' + \
                f'({self.dataset_pickle_path}).')
            json_df = pd.read_json(self.dataset_json_path)
            json_df.to_pickle(self.dataset_pickle_path)

        # Load dataset.
        self.df = pd.read_pickle(self.dataset_pickle_path)

    @property
    def dir_path(self):
        return f'{self.vertical.dir_path}/{self.name}'

    @property
    def dataset_json_path(self):
        return f'{self.dir_path}/dataset.json'

    @property
    def dataset_pickle_path(self):
        return f'{self.dir_path}/dataset.pkl'
