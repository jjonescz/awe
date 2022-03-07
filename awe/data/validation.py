import warnings

from tqdm.auto import tqdm

import awe.data.set.pages


def validate(pages: list[awe.data.set.pages.Page]):
    """Checks that label key-value pairs are consistent."""

    for page in tqdm(pages, desc='pages'):
        page: awe.data.set.pages.Page
        page_labels = page.get_labels()
        for key in page_labels.label_keys:
            values = page_labels.get_label_values(key)
            nodes = page_labels.get_labeled_nodes(key)
            expected = len(values)
            actual = len(nodes)
            if actual < expected:
                warnings.warn(
                    f'Found {actual} < {expected} nodes labeled ' + \
                    f'{repr(key)}={repr(values)} ({page.html_path}).')
