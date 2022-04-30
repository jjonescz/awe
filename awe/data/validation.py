import dataclasses
import warnings
from io import TextIOWrapper
from typing import Optional

from tqdm.auto import tqdm

import awe.data.html_utils
import awe.data.set.pages


@dataclasses.dataclass
class Validator:
    """Validates pages from a dataset."""

    labels: bool = True
    """Validate that labels are consistent."""

    only_cached_dom: bool = False
    """Skip pages that do not have their DOM loaded in memory."""

    visuals: bool = True
    """Validate that visuals are consistent."""

    labeled_boxes: bool = True
    """Validate that all target nodes have a bounding box (ergo are visible)."""

    num_invalid: int = 0
    """Count of invalid pages."""

    num_tested: int = 0
    """Count of validated pages (does not include skipped pages)."""

    file: Optional[TextIOWrapper] = None
    """File where invalid page paths are written to."""

    def summary(self):
        """Summarizes validation statistics into a dictionary."""

        stats = {
            'invalid': self.num_invalid,
            'tested': self.num_tested,
        }
        return {
            k: v for k, v in stats.items()
            if k == 'invalid' or v != 0
        }

    def summary_str(self):
        """Stringifies `summary`."""

        return ', '.join(f'{k}={v}' for k, v in self.summary().items())

    def write_invalid_to(self, file_path: str):
        """Opens file at `file_path` and sets it as the output `file`."""

        # pylint: disable-next=consider-using-with
        self.file = open(file_path, mode='w', encoding='utf-8')

    def validate_pages(self,
        pages: list[awe.data.set.pages.Page],
        progress_bar: Optional[str] = 'pages',
        max_invalid: Optional[int] = None
    ):
        """
        Validates the given set of `pages`.

        Parameters:
        - `progress_bar`: name of progress bar to show (or `None` to hide it),
        - `max_invalid`: stop the validation after this many invalid pages.
        """

        # Reset validation state.
        for page in pages:
            page.valid = None

        p = tqdm(pages, desc=progress_bar) if progress_bar is not None else None
        for page in (p or pages):
            if p is not None:
                p.set_postfix(self.summary(), refresh=False)
            self.validate_page(page)
            self.num_tested += 1
            if page.valid is False:
                self.num_invalid += 1
                if self.file is not None:
                    self.file.write(f'{page.original_html_path}\n')
            if max_invalid is not None and self.num_invalid >= max_invalid:
                break

    @staticmethod
    def get_selector_str(page: awe.data.set.pages.Page, key: str):
        """Gets representation of CSS selector for attribute `key` in `page`."""

        selector = page.labels.get_selector(key)
        s = '' if selector is None else f'({selector=})'
        return f'{key!r}{s}'

    def validate_page(self, page: awe.data.set.pages.Page):
        """Validates one `page`."""

        if self.only_cached_dom:
            page_dom = page.try_get_dom()
        else:
            page_dom = page.dom

        # Check that label key-value pairs are consistent.
        if self.labels:
            total = 0
            for key in page.labels.label_keys:
                values = page.labels.get_label_values(key)
                nodes = page.labels.get_labeled_nodes(key)
                expected = len(values)
                actual = len(nodes)
                total += actual
                if actual < expected:
                    page.valid = False
                    warnings.warn(
                        f'Found {actual} < {expected} nodes labeled ' +
                        f'{self.get_selector_str(page, key)}={values!r} ' +
                        f'({page.html_path!r}).')

                # Check that labeled nodes are not empty.
                for node in nodes:
                    if awe.data.html_utils.is_empty(node):
                        page.valid = False
                        xpath = awe.data.html_utils.get_xpath(node)
                        warnings.warn(
                            f'Node {xpath!r} labeled ' +
                            f'{self.get_selector_str(page, key)} is empty ' +
                            f'({page.html_path!r}).')

            if total == 0:
                page.valid = False
                warnings.warn(f'Nothing labeled in page {page.html_path!r}.')

            # Check that one node has only one label.
            if page_dom is not None:
                if page_dom.root is None:
                    page_dom.init_nodes()
                    page_dom.init_labels()
                for key, labeled_groups in page_dom.labeled_nodes.items():
                    for labeled_nodes in labeled_groups:
                        for node in labeled_nodes:
                            if len(node.label_keys) != 1:
                                page.valid = False
                                warnings.warn(
                                    f'Node {node.get_xpath()!r} has more than ' +
                                    f'one label key {node.label_keys!r} ' +
                                    f'({page.html_path!r}).')

        if self.visuals:
            if page_dom.root is None:
                page_dom.init_nodes()

            try:
                page_visuals = page.load_visuals()
            except FileNotFoundError as e:
                page.valid = False
                warnings.warn(f'No visuals for page {page.html_path!r}: {e}.')
                return

            # Check that extracted visual DOM has the same structure as parsed
            # DOM.
            try:
                page_visuals.fill_tree(page_dom)
            except RuntimeError as e:
                page.valid = False
                warnings.warn(
                    f'Cannot fill page visuals ({page.html_path!r}): {e}')
                return

            # Check that all target nodes have a bounding box.
            if self.labeled_boxes:
                page_dom.init_labels(propagate_to_leaves=False)
                for label_key, labeled_groups in page_dom.labeled_nodes.items():
                    for labeled_nodes in labeled_groups:
                        for n in labeled_nodes:
                            if n.box is None:
                                page.valid = False
                                warnings.warn(
                                    f'Node {n.get_xpath()!r} labeled ' +
                                    f'{self.get_selector_str(page, label_key)} ' +
                                    f'(among {len(labeled_nodes)}) has no ' +
                                    f'bounding box ({page.html_path!r}).')

        if page.valid is None:
            page.valid = True
