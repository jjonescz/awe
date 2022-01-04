import importlib

import torch
import numpy as np

from awe import utils, filtering, features, html_utils, awe_graph, visual
from awe.data import swde, live, dataset
from awe.features import extraction

for module in [utils, filtering, dataset, swde, live, features, extraction, html_utils, awe_graph, visual]:
    importlib.reload(module)

np.random.seed(42)
torch.manual_seed(42)

sds = swde.Dataset(suffix='-exact')

SUBSET = slice(2)
vertical = sds.verticals[0]
train_pages = vertical.websites[0].pages[:400] + vertical.websites[1].pages[:100] + vertical.websites[2].pages[:100]
val_pages = vertical.websites[3].pages[:100]
ds = dataset.DatasetCollection()
ds.create('train', train_pages[SUBSET], shuffle=True)
ds.create('val', val_pages[SUBSET])

ds.features = [
    features.Depth(),
    features.IsLeaf(),
    features.CharCategories(),
    features.Visuals(),
    features.CharIdentifiers(),
    features.WordIdentifiers(),
]

ds.create_dataloaders(batch_size=4)

for batch in ds['train'].loader:
    print(batch)
    break
