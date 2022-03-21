# Run: `python -m awe.inference`

import json
import sys

import awe.data.set.live
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
        try:
            data = json.loads(line)
            url = data['url']
            html_text = data['html']
            visuals = data['visuals']
            page = awe.data.set.live.Page(
                index=0,
                url=url,
                html_text=html_text,
                visuals_data=visuals
            )
            run = trainer.create_run([page], desc='live')
            preds = trainer.predict(run)
            df = trainer.decode(preds)
            response = {
                k: v
                for k, v in df.iloc[0].items()
            }
        except RuntimeError as e:
            response = { 'error': repr(e) }
        json.dump(response, sys.stdout)
        print() # commit the message by sending a newline character

if __name__ == '__main__':
    main()
