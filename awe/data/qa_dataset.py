import collections
import json

import pandas as pd
import selectolax.parser
from black import os
from tqdm.auto import tqdm

from awe import awe_graph


class QaDataset:
    """Dataset for question answering."""

    def __init__(self,
        name: str,
        df: pd.DataFrame
    ):
        self.name = name
        self.df = df

    def __getitem__(self, idx: int):
        return self.df.iloc[idx]

    def __len__(self):
        return len(self.df)

def prepare_dataset(pages: list[awe_graph.HtmlPage]):
    """Saves page texts to disk so that `QaDataset` can load them on demand."""

    with tqdm(desc='pages', total=len(pages)) as progress:
        skipped = 0

        # Group by folder.
        folders: dict[str, list[awe_graph.HtmlPage]] = \
            collections.defaultdict(list)
        for page in pages:
            folder = os.path.dirname(page.data_point_path)
            folders[folder].append(page)

        for folder, files in folders.items():
            # Load existing dataframe.
            df_path = os.path.join(folder, 'qa.csv')
            if os.path.exists(df_path):
                df = pd.read_csv(df_path, index_col=False)
            else:
                df = pd.DataFrame(columns=['id', 'text', 'labels'])

            # Add pages.
            new_data = {'id': [], 'text': [], 'labels': []}
            for page in files:
                # Skip existing.
                if (df['id'] == page.identifier).any():
                    skipped += 1
                    progress.set_postfix({ 'skipped': skipped }, refresh=False)
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
            df = df.append(pd.DataFrame(new_data))

            # Save dataframe.
            df.to_csv(df_path, index=False)

def extract_text(page: awe_graph.HtmlPage):
    """Converts page's HTML to text."""

    # pylint: disable-next=c-extension-no-member
    tree = selectolax.parser.HTMLParser(page.html)

    # Ignore some tags.
    for tag in ['script', 'style', 'head', '[document]', 'noscript', 'iframe']:
        for element in tree.css(tag):
            element.decompose()

    return tree.body.text(separator='\n')
