import collections
import dataclasses
import json
import re

import pandas as pd
import selectolax.parser
from black import os
from tqdm.auto import tqdm

from awe import awe_graph, utils

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

class QaDataset:
    """Dataset for question answering."""

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

def prepare_dataset(pages: list[awe_graph.HtmlPage], *,
    skip_existing: bool = True
):
    """Saves page texts to disk so that `QaDataset` can load them on demand."""

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
                text = extract_text(page)
                labels = {
                    f: page.get_groundtruth_texts(f)
                    for f in page.fields
                }
                new_data_idx.append(page.identifier)
                new_data['text'].append(text)
                new_data['labels'].append(json.dumps(labels))
                progress.update(1)

            # Append data.
            if len(new_data_idx) != 0:
                df.update(pd.DataFrame(new_data, index=new_data_idx))

                # Save dataframe.
                df.to_csv(df_path, index_label='id')

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

    # Collapse whitespace.
    text = re.sub(WHITESPACE_REGEX, ' ', text)

    return text
