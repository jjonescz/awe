from tqdm.auto import tqdm

import awe.data.graph.pages


def validate(pages: list[awe.data.graph.pages.Page]):
    """Checks that label key-value pairs are consistent."""

    for page in tqdm(pages, desc='pages'):
        page: awe.data.graph.pages.Page
        page_labels = page.get_labels()
        for key in page_labels.label_keys:
            values = page_labels.get_label_values(key)
            nodes = page_labels.get_labeled_nodes(key)
            expected = len(values)
            actual = len(nodes)
            if actual < expected:
                raise RuntimeError(
                    f'Found {actual} < {expected} nodes labeled ' + \
                    f'{repr(key)}={repr(values)} ({page.html_path}).')
