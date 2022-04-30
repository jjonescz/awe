"""A simple dataset implementation for live (unlabeled) pages."""

import dataclasses
from typing import Optional

import awe.data.parsing
import awe.data.set.labels
import awe.data.set.pages
import awe.model.classifier
import awe.training.trainer


@dataclasses.dataclass(eq=False)
class Page(awe.data.set.pages.Page):
    def __init__(self,
        index: int,
        url: str,
        html_text: str,
        visuals_data: dict[str],
        screenshot: Optional[bytes] = None
    ):
        super().__init__(website=None, index=index)
        self._url = url
        self.html_text = html_text
        self.visuals_data = visuals_data
        self._screenshot = screenshot
        self._labels = PageLabels(self)

    @property
    def file_name_no_extension(self):
        return self.url

    @property
    def dir_path(self):
        return '/LIVE'

    @property
    def url(self):
        return self._url

    @property
    def screenshot_bytes(self):
        return self._screenshot

    @property
    def labels(self):
        # This is overridden to avoid caching the labels (the default behavior
        # in the base class).
        return self._labels

    @property
    def index_in_dataset(self):
        return self.index

    def load_visuals(self):
        visuals = self.create_visuals()
        visuals.data = self.visuals_data
        return visuals

    def get_html_text(self):
        return self.html_text

    def get_labels(self):
        return self._labels

    def fill_labels(self,
        trainer: awe.training.trainer.Trainer,
        preds: list[awe.model.classifier.Prediction]
    ):
        self._labels = PredictedLabels(page=self, trainer=trainer, preds=preds)

class PageLabels(awe.data.set.labels.PageLabels):
    """Empty labels."""

    page: Page

    @property
    def label_keys(self):
        return []

    def get_label_values(self, _: str):
        return []

    def get_labeled_nodes(self, _: str):
        return []

class PredictedLabels(awe.data.set.labels.PageLabels):
    """
    Predicted labels that act as "gold" labels, so some code can be easily
    abstracted, e.g., `awe.data.visual.exploration`.
    """

    page: Page

    def __init__(self,
        page: Page,
        trainer: awe.training.trainer.Trainer,
        preds: list[awe.model.classifier.Prediction]
    ):
        super().__init__(page=page)
        self.trainer = trainer
        self.preds = preds

    @property
    def label_keys(self):
        return self.trainer.label_map.label_to_id.keys()

    def get_label_values(self, label_key: str):
        return [
            awe.data.parsing.normalize_node_text(n.text())
            for n in self.get_labeled_nodes(label_key)
        ]

    def get_labeled_nodes(self, label_key: str):
        decoded = self.trainer.decode_raw(self.preds)
        if len(decoded) == 0:
            return []
        node_preds = decoded[0][label_key]
        return [n.node.find_node().parsed for n in node_preds]
