from typing import TYPE_CHECKING

import torch

import awe.features.feature
import awe.data.graph.dom

if TYPE_CHECKING:
    import awe.model.classifier


class HtmlTag(awe.features.feature.Feature):
    html_tags: set[str]
    html_tag_ids: dict[str, int]

    def prepare(self, node: awe.data.graph.dom.Node, train: bool):
        if train:
            self.html_tags.add(node.html_tag)

    def initialize(self):
        # Map all found HTML tags to numbers. Note that 0 is reserved for
        # "unknown" tags.
        self.html_tag_ids = {
            c: i + 1
            for i, c in enumerate(self.html_tags)
        }

    def compute(self, batch: 'awe.model.classifier.ModelInput'):
        return torch.tensor(
            [
                self.html_tag_ids.get(node.html_tag, 0)
                for node in batch
            ],
            device=self.trainer.device
        )
