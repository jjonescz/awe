import pandas as pd
import torch
import transformers

import awe.qa.model
import awe.qa.postprocess


class Decoder:
    def __init__(self,
        tokenizer: transformers.PreTrainedTokenizerBase
    ):
        self.tokenizer = tokenizer

    def decode_predictions(self, predictions: list[awe.qa.model.Prediction]):
        return pd.DataFrame(self._iterate_decode_predictions(predictions))

    def _iterate_decode_predictions(self,
        predictions: list[awe.qa.model.Prediction]
    ):
        for pred in predictions:
            for row in range(pred.batch.input_ids.shape[0]):
                input_ids: torch.LongTensor = pred.batch.input_ids[row]

                question = self.tokenizer.decode(
                    input_ids[pred.batch.question_mask[row]]
                )

                gold_start = pred.batch['start_positions'][row]
                gold_end = pred.batch['end_positions'][row]
                gold_answer = self.tokenizer.decode(
                    input_ids[gold_start:gold_end + 1]
                )

                pred_start = torch.argmax(pred.outputs.start_logits[row])
                pred_end = torch.argmax(pred.outputs.end_logits[row])
                pred_answer = self.tokenizer.decode(
                    input_ids[pred_start:pred_end + 1]
                )

                post_span = awe.qa.postprocess.clamp_span(
                    pred_start, pred_end, pred.batch, row
                )

                yield {
                    'question': question,
                    'gold_span': (gold_start.item(), gold_end.item()),
                    'pred_span': (pred_start.item(), pred_end.item()),
                    'post_span': post_span,
                    'gold_answer': gold_answer,
                    'pred_answer': pred_answer,
                }
