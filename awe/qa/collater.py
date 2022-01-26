import warnings
from typing import Iterable, Optional

import transformers

import awe.qa.parser
from awe import awe_graph


# Inspired by https://huggingface.co/transformers/v3.2.0/custom_datasets.html.
class QaCollater:
    label_to_question: Optional[dict[str, str]] = None

    def __init__(self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        max_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, pages: list[awe_graph.HtmlPage]):
        return self.get_encodings(pages).convert_to_tensors('pt')

    def get_encodings(self, pages: list[awe_graph.HtmlPage]):
        # Expand each page into multiple training samples (one question-answer
        # per label).
        label_maps = [awe.qa.parser.get_page_labels(page) for page in pages]
        samples = [
            (page, label, values)
            for page, label_map in zip(pages, label_maps)
            for label, values in label_map.items()
            if len(values) != 0
        ]

        # Tokenize batch.
        questions = [[self.get_question(label)] for _, label, _ in samples]
        texts = [awe.qa.parser.get_page_words(page) for page, _, _ in samples]
        encodings = self.tokenizer(questions, texts,
            is_split_into_words=True,
            truncation=True,
            padding='max_length' if self.max_length is not None else True,
            max_length=self.max_length,
        )

        # Find start/end positions.
        start_positions = []
        end_positions = []
        for (_, _, values), words in zip(samples, texts):
            spans = [
                span
                for value in values
                for span in awe.qa.parser.get_spans(words, value)
            ]
            start_positions.append(self.spans_to_positions(
                encodings, (start for start, _ in spans)
            ))
            end_positions.append(self.spans_to_positions(
                encodings, (end - 1 for _, end in spans)
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
        return f'What is the {label}?'

    def spans_to_positions(self,
        encodings: transformers.BatchEncoding,
        spans: Iterable[int]
    ) -> int:
        positions = [
            encodings.char_to_token(i, sequence_index=1)
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
