import collections
import dataclasses
import json

import pandas as pd
import selectolax.parser
from black import os
from tqdm.auto import tqdm

from awe import awe_graph

@dataclasses.dataclass
class QaEntry:
    id: str
    text: str
    labels: dict[str, list[str]]

class QaDataset:
    """Dataset for question answering."""

    def __init__(self, pages: list[awe_graph.HtmlPage]):
        self.pages = pages
        self.dfs: dict[str, pd.DataFrame] = {}

    def __getitem__(self, idx: int):
        page = self.pages[idx]
        folder = os.path.dirname(page.data_point_path)
        df = self.get_df(folder)
        row = df[df['id'] == page.identifier].iloc[0]
        labels = json.loads(row['labels'])
        return QaEntry(page.identifier, row['text'], labels)

    def __len__(self):
        return len(self.pages)

    def get_df(self, folder: str):
        df = self.dfs.get(folder)
        if df is None:
            _, df = load_dataframe(folder)
            self.dfs[folder] = df
        return df

def prepare_dataset(pages: list[awe_graph.HtmlPage]):
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
            new_data = {'id': [], 'text': [], 'labels': []}
            for page in files:
                # Skip existing.
                if (df['id'] == page.identifier).any():
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
                new_data['id'].append(page.identifier)
                new_data['text'].append(text)
                new_data['labels'].append(json.dumps(labels))
                progress.update(1)

            # Append data.
            if len(new_data['id']) != 0:
                df = df.append(pd.DataFrame(new_data))

                # Save dataframe.
                df.to_csv(df_path, index=False)

def load_dataframe(folder: str):
    df_path = os.path.join(folder, 'qa.csv')
    if os.path.exists(df_path):
        return df_path, pd.read_csv(df_path, index_col=False)
    return df_path, pd.DataFrame(columns=['id', 'text', 'labels'])

def extract_text(page: awe_graph.HtmlPage):
    """Converts page's HTML to text."""

    # pylint: disable-next=c-extension-no-member
    tree = selectolax.parser.HTMLParser(page.html)

    # Ignore some tags.
    for tag in ['script', 'style', 'head', '[document]', 'noscript', 'iframe']:
        for element in tree.css(tag):
            element.decompose()

    return tree.body.text(separator='\n')
