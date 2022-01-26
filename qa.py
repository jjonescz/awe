import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data

from awe import gym, qa_model
from awe.data import qa_dataset, swde

PREPARE_DATA = True
TRAIN_SUBSET = 2000
VAL_SUBSET = 50

def main():
    # Load websites from one vertical.
    sds = swde.Dataset(suffix='-exact')
    websites = sds.verticals[0].websites

    # Split websites.
    train_website_indices = [0, 3, 4, 5, 7]
    val_website_indices = [i
        for i in range(len(websites))
        if i not in train_website_indices
    ]
    train_websites = [websites[i] for i in train_website_indices]
    val_websites = [websites[i] for i in val_website_indices]
    train_website_names = [w.name for w in train_websites]
    val_website_names = [w.name for w in val_websites]
    print(f'{train_website_names=}, {val_website_names=}')

    # Take pages.
    train_pages = [p for w in train_websites for p in w.pages]
    val_pages = [p for w in val_websites for p in w.pages]
    print(f'{len(train_pages)=}, {len(val_pages)=}')

    # Take subset.
    rng = np.random.default_rng(42)
    train_pages = rng.choice(train_pages, TRAIN_SUBSET, replace=False)
    val_pages = rng.choice(val_pages, VAL_SUBSET, replace=False)
    print(f'{len(train_pages)=}, {len(val_pages)=}')

    # Prepare data.
    if PREPARE_DATA:
        qa_dataset.prepare_entries(train_pages + val_pages)

    # Load pre-trained models.
    pipeline = qa_model.QaPipeline()
    pipeline.load()
    model = qa_model.QaModel(pipeline)

    # Prepare dataloaders.
    train_ds = qa_dataset.QaTorchDataset(train_pages, pipeline.tokenizer)
    val_ds = qa_dataset.QaTorchDataset(val_pages, pipeline.tokenizer)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1)

    # Fine-tune.
    g = gym.Gym(None, None, version_name='')
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=1,
        logger=g.create_logger(),
    )
    trainer.fit(model, train_loader, val_loader)

    # Validate.
    trainer.validate(model, val_loader)

if __name__ == '__main__':
    main()
