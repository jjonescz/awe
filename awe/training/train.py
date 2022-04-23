"""
Script for training and validation.

Usage:
1. Set parameters in `data/params.json`.
2. Run `python -m awe.training.train`.
"""

import awe.training.params
import awe.training.trainer


def main():
    params = awe.training.params.Params.load_user(normalize=True)
    if params is None:
        print('Revisit params and re-run.')
        return
    print(f'{params=}')
    trainer = awe.training.trainer.Trainer(params)
    trainer.load_dataset()
    trainer.init_features()
    trainer.split_data()
    trainer.create_dataloaders()
    trainer.create_model()
    trainer.create_version()
    trainer.train()

if __name__ == '__main__':
    main()
