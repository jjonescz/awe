# Run: `python -m awe.inference`

import json
import sys

import awe.training.logging
import awe.training.params
import awe.training.trainer


def main():
    version = awe.training.logging.Version.get_latest()
    checkpoint = version.get_checkpoints()[-1]
    params = awe.training.params.Params.load_version(version)
    print(f'{params=}')
    trainer = awe.training.trainer.Trainer(params)
    trainer.load_pretrained()
    trainer.init_features()
    trainer.restore_checkpoint(checkpoint)
    trainer.restore_features()
    trainer.create_model()
    trainer.restore_model()

    print('Inference started.')

    for line in sys.stdin:
        data = json.loads(line)
        json.dump({ 'got': data }, sys.stdout)
        print() # commit the message by sending a newline character

if __name__ == '__main__':
    main()
