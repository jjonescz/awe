import pytorch_lightning as pl

from awe.data import dataset

class DataModule(pl.LightningDataModule):
    def __init__(self, ds: dataset.DatasetCollection):
        super().__init__()
        self.ds = ds

    def train_dataloader(self):
        return self.ds['train'].loader

    def val_dataloader(self):
        return self._prefixed_dataloader('val')

    def test_dataloader(self):
        return self._prefixed_dataloader('test')

    def _prefixed_dataloader(self, prefix: str):
        return [
            d.loader
            for name, d in self.ds.datasets.items()
            if name.startswith(prefix)
        ]
