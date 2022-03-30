import dataclasses
import warnings
from io import TextIOWrapper
from typing import Optional

from tqdm.auto import tqdm

import awe.data.html_utils
import awe.data.set.pages


@dataclasses.dataclass
class Validator:
    labels: bool = True
    visuals: bool = True
    num_invalid: int = 0
    file: Optional[TextIOWrapper] = None

    def write_invalid_to(self, file_path: str):
        self.file = open(file_path, mode='w', encoding='utf-8')

    def validate_pages(self,
        pages: list[awe.data.set.pages.Page],
        progress_bar: Optional[str] = 'pages',
        max_invalid: Optional[int] = None
    ):
        # Reset validation state.
        for page in pages:
            page.valid = None

        self.num_invalid = 0
        for page in tqdm(pages, desc=progress_bar) if progress_bar is not None else pages:
            self.validate_page(page)
            if page.valid is False:
                self.num_invalid += 1
                if self.file is not None:
                    self.file.write(f'{page.original_html_path}\n')
            if max_invalid is not None and self.num_invalid >= max_invalid:
                break

    def validate_page(self, page: awe.data.set.pages.Page):
        # Check that label key-value pairs are consistent.
        if self.labels:
            page_labels = page.get_labels()
            for key in page_labels.label_keys:
                values = page_labels.get_label_values(key)
                nodes = page_labels.get_labeled_nodes(key)
                expected = len(values)
                actual = len(nodes)
                if actual < expected:
                    page.valid = False
                    warnings.warn(
                        f'Found {actual} < {expected} nodes labeled ' +
                        f'{key!r}={values!r} ({page.html_path!r}).')

                # Check that labeled nodes are not empty.
                for node in nodes:
                    if node.child is None and not node.text(deep=False):
                        page.valid = False
                        xpath = awe.data.html_utils.get_xpath(node)
                        warnings.warn(
                            f'Node {xpath!r} labeled {key!r} is empty ' +
                            f'({page.html_path!r}).')

        # Check that extracted visual DOM has the same structure as parsed DOM.
        if self.visuals:
            page_dom = page.dom
            page_dom.init_nodes()
            page_visuals = page.load_visuals()
            try:
                page_visuals.fill_tree(page_dom)
            except RuntimeError as e:
                page.valid = False
                warnings.warn(
                    f'Cannot fill page visuals ({page.html_path!r}): {e}')

        if page.valid is None:
            page.valid = True
