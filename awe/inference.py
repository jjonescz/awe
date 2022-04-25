"""
Script for loading the model and running as a server doing inference on demand.

Used by TypeScript demo application.

Run as `python -m awe.inference`.
"""

import base64
import io
import json
import sys
import traceback

import awe.data.graph.pred
import awe.data.parsing
import awe.data.set.live
import awe.data.visual.exploration
import awe.training.params
import awe.training.trainer
import awe.training.versioning


def main():
    version = awe.training.versioning.Version.get_latest()
    checkpoint = version.get_checkpoints()[-1]
    params = awe.training.params.Params.load_version(version)
    params.patch_for_inference()
    print(f'{params=}')
    trainer = awe.training.trainer.Trainer(params)
    trainer.init_features()
    trainer.restore_checkpoint(checkpoint)
    trainer.restore_features()
    trainer.create_model()
    trainer.restore_model()

    # IMPORTANT: This line is also used as a clue to let TypeScript client node
    # the inference server is ready.
    print('Inference started.')

    # Read "requests", run inference, and write "responses".
    for line in sys.stdin:
        try:
            data = json.loads(line)
            url = data['url']
            html_text = data['html']
            visuals = data['visuals']
            screenshot = base64.b64decode(data['screenshot'])
            page = awe.data.set.live.Page(
                index=0,
                url=url,
                html_text=html_text,
                visuals_data=visuals,
                screenshot=screenshot,
            )
            run = trainer.create_run([page], desc='live')
            preds = trainer.predict(run)
            decoded = trainer.decode_raw(preds)

            # Render screenshot with predicted nodes highlighted.
            page.fill_labels(trainer, preds)
            fig = awe.data.visual.exploration.plot_pages([(page,)],
                set_title=False,
                crop=False,
            )
            with io.BytesIO() as out_bytes:
                fig.savefig(out_bytes, format='png', bbox_inches='tight')
                out_b64 = base64.b64encode(out_bytes.getvalue()).decode('ascii')

            response = {
                'pages': [
                    {
                        k: [serialize_prediction(p) for p in v]
                        for k, v in d.items()
                    }
                    for d in decoded
                ],
                'screenshot': out_b64
            }
        except RuntimeError:
            response = { 'error': traceback.format_exc() }
        json.dump(response, sys.stdout)
        print() # commit the message by sending a newline character

def serialize_prediction(p: awe.data.graph.pred.NodePrediction):
    node = p.node.find_node()
    return {
        'text': (
            awe.data.parsing.normalize_node_text(node.text)
            if node.is_text else None
        ),
        'url': node.get_attribute('href'),
        'xpath': node.get_xpath(),
        'confidence': p.confidence,
        'probability': p.probability
    }

if __name__ == '__main__':
    main()
