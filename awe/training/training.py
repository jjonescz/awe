# 1. Set parameters in `data/params.json`.
# 2. Run `python -m awe.training.training`.

import awe.training.params
import awe.training.trainer

def main():
    params = awe.training.params.Params.load_user(normalize=True)
    if params is None:
        return
    print(f'{params=}')
    trainer = awe.training.trainer.Trainer(params)
    trainer.load_pretrained()
    trainer.load_dataset()
    trainer.prepare_features()
    trainer.create_model()
    trainer.create_version()
    trainer.train()

if __name__ == '__main__':
    main()
