# 1. Set parameters in `data/qa-params.json`.
# 2. Run `python -m awe.qa.training`.

import awe.qa.trainer

def main():
    params = awe.qa.trainer.TrainerParams.load_user()
    if params is None:
        return
    print(f'{params=}')
    trainer = awe.qa.trainer.Trainer(params)
    trainer.create_version()
    trainer.create_writer()
    trainer.load_pipeline()
    trainer.load_data()
    trainer.create_model()
    trainer.train()

if __name__ == '__main__':
    main()
