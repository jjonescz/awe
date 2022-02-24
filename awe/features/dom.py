from typing import TYPE_CHECKING

import torch

import awe.features.feature
import awe.data.graph.dom

if TYPE_CHECKING:
    import awe.model.classifier


class HtmlTag(awe.features.feature.Feature):

    def prepare(self, node: awe.data.graph.dom.Node):
        self.trainer.extractor.context.html_tags.add(node.html_tag)

    def initialize(self):
        context = self.trainer.extractor.context

        # Map all found HTML tags to numbers. Note that 0 is reserved for
        # "unknown" tags.
        context.html_tag_ids = {
            c: i + 1
            for i, c in enumerate(context.html_tags)
        }

    def compute(self, batch: 'awe.model.classifier.ModelInput'):
        context = self.trainer.extractor.context

        return torch.tensor(
            [
                context.html_tag_ids.get(node.html_tag, 0)
                for node in batch
            ],
            device=self.trainer.device
        )
