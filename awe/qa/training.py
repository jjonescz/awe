# Run: `python -m awe.qa.training`

import awe.qa.trainer

def main():
    params = awe.qa.trainer.QaTrainerParams(
        batch_size=1,
        train_subset=1000,
        version_name='qa-train-1000'
    )
    trainer = awe.qa.trainer.QaTrainer(params)
    trainer.load_pipeline()
    trainer.load_data()
    trainer.prepare_data()
    trainer.create_model()
    trainer.delete_previous_version()
    trainer.train()

if __name__ == '__main__':
    main()
