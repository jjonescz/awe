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
import awe.model.classifier
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
    trainer.extractor.enable_cache(False)

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
            preds = postprocess(preds)

            # Render screenshot with predicted nodes highlighted.
            page.fill_labels(trainer, preds)
            fig = plot_screenshot(page)
            with io.BytesIO() as out_bytes:
                fig.savefig(out_bytes, format='png', bbox_inches='tight')
                out_b64 = base64.b64encode(out_bytes.getvalue()).decode('ascii')

            response = {
                'pages': [
                    {
                        k: [serialize_prediction(p) for p in v]
                        for k, v in d.items()
                    }
                    for d in trainer.decode_raw(preds)
                ],
                'screenshot': out_b64
            }
        except RuntimeError:
            response = { 'error': traceback.format_exc() }
        json.dump(response, sys.stdout)
        print() # commit the message by sending a newline character

def postprocess(preds: list[awe.model.classifier.Prediction]):
    """
    Performs simple post-processing.

    Only nodes that are inside the screen are included.
    """

    return [
        pred.filter_nodes(lambda n: n.box is None or n.box.is_positive)
        for pred in preds
    ]

def plot_screenshot(page: awe.data.set.live.Page):
    explorer = awe.data.visual.exploration.PageExplorer(page,
        crop=False,
        init_page=False,
    )
    return awe.data.visual.exploration.plot_explorers([(explorer,)],
        set_title=False,
    )

def serialize_prediction(p: awe.data.graph.pred.NodePrediction):
    """
    Converts Python object `NodePrediction` to a JSON object compatible with
    TypeScript interface `NodePrediction` defined in `js/lib/demo/python.ts`.
    """

    node = p.node.find_node()
    return {
        'text': (
            awe.data.parsing.normalize_node_text(node.text)
            if node.is_text else None
        ),
        'url': node.get_attribute('src'),
        'xpath': node.get_xpath(),
        'confidence': p.confidence,
        'probability': p.probability,
        'box': node.box.as_tuple() if node.box else None,
    }

if __name__ == '__main__':
    main()
