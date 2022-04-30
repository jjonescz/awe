import math

import torch

import awe.features.feature
import awe.data.graph.dom
import awe.data.visual.structs


class HtmlTag(awe.features.feature.Feature):
    """HTML tag name feature."""

    _html_tags: set[str]
    html_tag_ids: dict[str, int]

    def get_pickled_keys(self):
        return ('html_tag_ids',)

    def __post_init__(self, restoring: bool):
        if not restoring:
            self._html_tags = set()

    def prepare(self, node: awe.data.graph.dom.Node, train: bool):
        node.semantic_html_tag = node.find_semantic_html_tag()

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

    def compute(self, batch: list[awe.data.graph.dom.Node]):
        return torch.tensor(
            [
                self.html_tag_ids.get(node.semantic_html_tag, 0)
                for node in batch
            ],
            device=self.trainer.device
        )

class Position(awe.features.feature.Feature):
    """Visual position feature. See `compute_position`."""

    out_dim: int = 4

    def compute(self, batch: list[awe.data.graph.dom.Node]):
        return torch.tensor(
            [compute_position(n) for n in batch],
            device=self.trainer.device
        )

def compute_position(node: awe.data.graph.dom.Node):
    """
    Computes node's relative xy position and size.

    Given the bounding box of a node `[x, y, w, h]` and its page `[0, 0, W, H]`,
    this feature is computed as `[x/W, y/H, log(w/W), log(h/H)]`.
    """

    root_box = node.dom.root.box
    return (
        node.box.x / root_box.width,
        node.box.y / root_box.height,
        _safe_log(node.box.width / root_box.width),
        _safe_log(node.box.height / root_box.height),
    )

def _safe_log(x: float):
    if x > 0:
        return math.log(x)
    return 0
