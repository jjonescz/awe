import warnings
from typing import Iterable, Optional

import transformers

import awe.qa.parser
import awe.qa.sampler


# Inspired by https://huggingface.co/transformers/v3.2.0/custom_datasets.html.
class Collater:
    label_to_question: Optional[dict[str, str]] = None

    def __init__(self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        max_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples: list[awe.qa.sampler.Sample]):
        return self.get_encodings(samples).convert_to_tensors('pt')

    def get_encodings(self, samples: list[awe.qa.sampler.Sample]):
        # Tokenize batch.
        questions = [[self.get_question(sample.label)] for sample in samples]
        texts = [awe.qa.parser.get_page_words(sample.page) for sample in samples]
        encodings = self.tokenizer(questions, texts,
            is_split_into_words=True,
            truncation=True,
            padding='max_length' if self.max_length is not None else True,
            max_length=self.max_length,
        )

        # Find start/end positions.
        start_positions = []
        end_positions = []
        for idx, (sample, words) in enumerate(zip(samples, texts)):
            spans = [
                span
                for value in sample.values
                for span in awe.qa.parser.get_spans(words, value)
            ]
            start_positions.append(self.spans_to_positions(
                encodings, idx, (start for start, _ in spans)
            ))
            end_positions.append(self.spans_to_positions(
                encodings, idx, (end - 1 for _, end in spans)
            ))
        encodings['start_positions'] = start_positions
        encodings['end_positions'] = end_positions

        return encodings

    def get_question(self, label: str):
        if self.label_to_question is None:
            return self.get_default_question(label)
        question = self.label_to_question.get(label)
        if question is None:
            warnings.warn(f'No question mapping for label {repr(label)}')
            return self.get_default_question(label)
        return question

    def get_default_question(self, label: str):
        humanized_label = label.replace('_', ' ')
        return f'What is the {humanized_label}?'

    def spans_to_positions(self,
        encodings: transformers.BatchEncoding,
        batch_idx: int,
        spans: Iterable[int]
    ) -> int:
        positions = [
            encodings.char_to_token(batch_idx, i, sequence_index=1)
            # Handle when answer is truncated from the context.
            or self.tokenizer.model_max_length
            for i in spans
        ]
        if len(positions) == 0:
            # Return something even if there is no value for the label.
            positions = [self.tokenizer.model_max_length]
        # TODO: We limit the number of answers to one for now, because the model
        # head doesn't support more.
        return positions[0]
