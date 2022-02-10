# 1. Set parameters in `data/params.json`.
# 2. Run `python -m awe.training.script`.

import awe.training.params
import awe.training.trainer


def main():
    params = awe.training.params.Params.load_user()
    if params is None:
        return
    print(f'{params=}')
    trainer = awe.training.trainer.Trainer(params)
    trainer.create_version()
    trainer.create_writer()
    trainer.load_pretrained()
    trainer.load_data()
    trainer.create_model()
    trainer.train()

if __name__ == '__main__':
    main()
