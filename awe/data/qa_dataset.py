import collections
import dataclasses
import html
import json
import os
import re
import warnings
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import selectolax.parser
import torch
import transformers
from tqdm.auto import tqdm

from awe import awe_graph, qa_model, utils

WHITESPACE_REGEX = r'(\s|[\u200b])+'
"""Matches whitespace characters."""


@dataclasses.dataclass
class QaEntry:
    """Question answering example corresponding to one `HtmlPage`."""

    id: str
    """Corresponds to `HtmlPage.identity`."""

    text: str
    """Text extracted from the page's HTML."""

    labels: dict[str, list[str]]
    """Mapping from label classes to their values."""

    @staticmethod
    def create(page: awe_graph.HtmlPage):
        text = extract_text(page)
        labels = {
            # Note that labels may contain HTML entities (e.g., `&amp;`), that's
            # why we call `html.unescape` on them.
            f: [
                collapse_whitespace(html.unescape(v))
                for v in page.get_groundtruth_texts(f)
            ]
            for f in page.fields
        }
        return QaEntry(page.identifier, text, labels)

    def get_answer_spans(self, label: str):
        return [span
            for value in self.labels[label]
            for span in self.get_spans(value)
        ]

    def get_all_answer_spans(self):
        return {
            label: self.get_answer_spans(label)
            for label in self.labels.keys()
        }

    def get_spans(self, value: str):
        return [
            (start, start + len(value))
            for start in utils.find_all(self.text, value)
        ]

    def get_label_at(self, idx: int):
        return utils.at_index(self.labels.items(), idx)

    def validate(self):
        for label, values in self.labels.items():
            expected = len(values)
            actual = len(self.get_answer_spans(label))
            if actual < expected:
                plural = 's' if expected > 1 else ''
                values_str = ', '.join(f'"{value}"' for value in values)
                raise RuntimeError(f'Expected to find at least {expected} ' + \
                    f'value{plural} for label "{label}" ({values_str}) but ' + \
                    f'found {actual} ({self.id}).')

class QaEntryLoader:
    """Can load prepared `QaEntry`s for an `HtmlPage`."""

    def __init__(self, pages: list[awe_graph.HtmlPage]):
        self.pages = pages
        self.dfs: dict[str, pd.DataFrame] = {}

    def __getitem__(self, idx: int):
        page = self.pages[idx]
        return self.get_entry(page)

    def __len__(self):
        return len(self.pages)

    def get_df(self, folder: str):
        df = self.dfs.get(folder)
        if df is None:
            _, df = load_dataframe(folder)
            self.dfs[folder] = df
        return df

    def get_entry(self, page: awe_graph.HtmlPage):
        folder = os.path.dirname(page.data_point_path)
        df = self.get_df(folder)
        row = df[df.index == page.identifier].iloc[0]
        labels = json.loads(row['labels'])
        return QaEntry(page.identifier, row['text'], labels)

    def validate(self):
        for page in tqdm(self.pages, desc='pages'):
            self.get_entry(page).validate()

    @property
    def max_label_count(self):
        return max(len(p.fields) for p in self.pages)

    def prepare_entries(self,
        skip_existing: bool = True
    ):
        prepare_entries(self.pages, skip_existing=skip_existing)

# Inspired by https://huggingface.co/transformers/v3.2.0/custom_datasets.html.
class QaTorchDataset(torch.utils.data.Dataset):
    """PyTorch dataset for question answering."""

    loader: QaEntryLoader
    label_to_question: Optional[dict[str, str]] = None

    def __init__(self,
        loader_or_pages: Union[QaEntryLoader, list[awe_graph.HtmlPage]],
        tokenizer: transformers.PreTrainedTokenizerBase
    ):
        if hasattr(loader_or_pages, 'pages'):
            self.loader = loader_or_pages
        else:
            self.loader = QaEntryLoader(loader_or_pages)
        self.tokenizer = tokenizer

        label_counts = lambda: (len(p.fields) for p in self.loader.pages)
        min_label_count = min(label_counts())
        max_label_count = max(label_counts())
        assert min_label_count == max_label_count, 'Every page should have ' + \
            'the same number of label classes, but found range ' + \
            f'[{min_label_count}, {max_label_count}].'
        self.label_count = min_label_count

    def __getitem__(self, idx: int):
        return self.get_encodings(idx).convert_to_tensors('pt')

    def get_encodings(self, idx: int):
        entry, label, _ = self.at(idx)

        # Tokenize.
        question = self.get_question(label)
        encodings = self.tokenizer(question, entry.text,
            truncation=True,
            padding=True
        )

        # Find start/end positions.
        spans = entry.get_answer_spans(label)
        encodings['start_positions'] = self.spans_to_positions(
            encodings, (start for start, _ in spans)
        )
        encodings['end_positions'] = self.spans_to_positions(
            encodings, (end - 1 for _, end in spans)
        )

        return encodings

    def spans_to_positions(self,
        encodings: transformers.BatchEncoding,
        spans: Iterable[int]
    ):
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

    def __len__(self):
        return len(self.loader) * self.label_count

    def at(self, idx: int):
        # For each label, give separate question-answer example.
        entry = self.loader[idx // self.label_count]
        label_idx = self.to_label_idx(idx)
        label, values = entry.get_label_at(label_idx)
        return entry, label, values

    def to_label_idx(self, idx: int):
        return idx % self.label_count

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

    def decode(self, encodings):
        return [
            self.tokenizer.decode(encodings['input_ids'][0, start:end + 1])
            for start, end in zip(
                encodings['start_positions'],
                encodings['end_positions']
            )
        ]

    def validate(self, idx: int):
        entry, label, values = self.at(idx)
        encodings = self[idx]
        answers = self.decode(encodings)

        # Ignore repetition and spaces.
        values = set(remove_whitespace(v) for v in values)
        answers = set(remove_whitespace(a) for a in answers)

        if values != answers:
            warnings.warn(f'Inconsistency at {idx} ({entry.id}): ' + \
                f'{label=}, {values=}, {answers=}.')

    def validate_all(self):
        for i in tqdm(range(len(self))):
            self.validate(i)

    def decode_predictions(self, predictions: list[qa_model.QaModelPrediction]):
        return list(self._iterate_decode_predictions(predictions))

    def _iterate_decode_predictions(self,
        predictions: list[qa_model.QaModelPrediction]
    ):
        for pred in predictions:
            input_ids: torch.LongTensor = pred.batch['input_ids'][0, :]

            special_mask = np.array(self.tokenizer.get_special_tokens_mask(
                input_ids,
                already_has_special_tokens=True
            ))
            special_indices, = np.where(special_mask == 1)
            question = self.tokenizer.decode(
                input_ids[special_indices[0] + 1:special_indices[1]]
            )

            gold_start = pred.batch['start_positions'][0]
            gold_end = pred.batch['end_positions'][0]
            gold_answer = self.tokenizer.decode(
                input_ids[gold_start:gold_end + 1]
            )

            pred_start = torch.argmax(pred.outputs.start_logits)
            pred_end = torch.argmax(pred.outputs.end_logits)
            pred_answer = self.tokenizer.decode(
                input_ids[pred_start:pred_end + 1]
            )

            yield question, gold_answer, pred_answer

def prepare_entries(pages: list[awe_graph.HtmlPage], *,
    skip_existing: bool = True
):
    """
    Saves page texts to disk so that `QaEntryLoader` can load them on demand.
    """

    with tqdm(desc='pages', total=len(pages)) as progress:
        progress_data = collections.defaultdict(int)

        # Group by folder.
        folders: dict[str, list[awe_graph.HtmlPage]] = \
            collections.defaultdict(list)
        for page in pages:
            folder = os.path.dirname(page.data_point_path)
            folders[folder].append(page)

        for folder, files in folders.items():
            # Update progress bar.
            progress_data['folder'] = folder
            progress.set_postfix(progress_data)

            # Load existing dataframe.
            df_path, df = load_dataframe(folder)

            # Add pages.
            new_data_idx = []
            new_data = { 'text': [], 'labels': [] }
            for page in files:
                # Skip existing.
                if skip_existing and (df.index == page.identifier).any():
                    progress_data['skipped'] += 1
                    progress.set_postfix(progress_data, refresh=False)
                    progress.update(1)
                    continue

                # Process page.
                entry = QaEntry.create(page)
                new_data_idx.append(entry.id)
                new_data['text'].append(entry.text)
                new_data['labels'].append(json.dumps(entry.labels))
                progress.update(1)

            # Append data.
            if len(new_data_idx) != 0:
                new_df = pd.DataFrame(new_data, index=new_data_idx)
                df = df.combine_first(new_df)

                # Save dataframe.
                df.to_csv(df_path, index_label='id')
                print(f'Saved {df_path}')

def load_dataframe(folder: str):
    df_path = os.path.join(folder, 'qa.csv')
    if os.path.exists(df_path):
        return df_path, pd.read_csv(df_path, index_col='id')
    return df_path, pd.DataFrame(columns=['text', 'labels'])

def extract_text(page: awe_graph.HtmlPage):
    """Converts page's HTML to text."""

    # pylint: disable-next=c-extension-no-member
    tree = selectolax.parser.HTMLParser(page.html)

    # Ignore some tags.
    for tag in ['script', 'style', 'head', '[document]', 'noscript', 'iframe']:
        for element in tree.css(tag):
            element.decompose()

    text = tree.body.text(separator=' ')
    text = collapse_whitespace(text)
    return text

def collapse_whitespace(text: str):
    return re.sub(WHITESPACE_REGEX, ' ', text)

def remove_whitespace(text: str):
    return re.sub(WHITESPACE_REGEX, '', text)
