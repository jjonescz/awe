from typing import TYPE_CHECKING

import torch

import awe.features.feature
import awe.data.graph.dom
import awe.data.visual.structs
import awe.utils

if TYPE_CHECKING:
    import awe.model.classifier


class HtmlTag(awe.features.feature.Feature, awe.utils.PickleSubset):
    _html_tags: set[str]
    html_tag_ids: dict[str, int]

    def __post_init__(self, restoring: bool):
        if not restoring:
            self._html_tags = set()

    def get_pickled_keys(self):
        return ('html_tag_ids',)

    def prepare(self, node: awe.data.graph.dom.Node, train: bool):
        # Find most semantic HTML tag for the node.
        semantic = node.unwrap(tag_names={ 'span', 'div' })
        node.semantic_html_tag = semantic.html_tag

        if train:
            self._html_tags.add(node.semantic_html_tag)

    def freeze(self):
        # Map all found HTML tags to numbers. Note that 0 is reserved for
        # "unknown" tags.
        self.html_tag_ids = {
            c: i + 1
            for i, c in enumerate(self._html_tags)
        }
        self._html_tags = None

    def compute(self, batch: 'awe.model.classifier.ModelInput'):
        return torch.tensor(
            [
                self.html_tag_ids.get(node.semantic_html_tag, 0)
                for node in batch
            ],
            device=self.trainer.device
        )

class Position(awe.features.feature.Feature):
    root_box: awe.data.visual.structs.BoundingBox
    out_dim: int = 4

    def prepare(self, node: awe.data.graph.dom.Node, train: bool):
        if node.is_root:
            self.root_box = node.box

    def compute(self, batch: 'awe.model.classifier.ModelInput'):
        # For each node, compute its relative xy position and size.
        coords = torch.tensor(
            [(n.box.x, n.box.y) for n in batch],
            device=self.trainer.device
        ) # [N, 2]
        size = torch.tensor(
            [(n.box.width, n.box.height) for n in batch],
            device=self.trainer.device
        ) # [N, 2]
        rect = torch.tensor(
            [self.root_box.width, self.root_box.height],
            device=self.trainer.device
        ) # [2]
        return torch.cat((coords / rect, torch.log(size / rect)), dim=-1) # [N, 4]
