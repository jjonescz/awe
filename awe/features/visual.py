import torch

import awe.data.graph.dom
import awe.data.visual.attribute
import awe.data.visual.context
import awe.features.feature
import awe.utils


class Visuals(awe.features.feature.Feature):
    """
    Visual features.

    Forwards preparation/computation calls to each enabled visual attribute
    defined in module `awe.data.visual.attribute`.
    """

    extraction: awe.data.visual.context.Extraction
    out_dim: int = None

    def __post_init__(self, restoring: bool):
        # Filter visual attributes according to training params.
        enabled = self.trainer.params.enabled_visuals
        disabled = self.trainer.params.disabled_visuals
        self.visual_attributes = [
            a
            for a in awe.data.visual.attribute.VISUAL_ATTRIBUTES.values()
            if enabled is None or a.name in enabled
            if disabled is None or a.name not in disabled
        ]

        if not restoring:
            self.extraction = awe.data.visual.context.Extraction()
        else:
            self.freeze()

    def get_pickled_keys(self):
        return ('extraction',)

    def prepare(self, node: awe.data.graph.dom.Node, train: bool):
        if train and node.needs_visuals:
            for a in self.visual_attributes:
                a.prepare(awe.data.visual.attribute.AttributeContext(
                    node=node,
                    extraction=self.extraction,
                ))

    def freeze(self):
        self.extraction.freeze()
        self.out_dim = sum(
            a.get_out_dim(self.extraction)
            for a in self.visual_attributes
        )

    def compute(self, batch: list[awe.data.graph.dom.Node]):
        return torch.tensor(
            [
                self._compute_one(node)
                for node in batch
            ],
            dtype=torch.float32,
            device=self.trainer.device,
        )

    def _compute_one(self, node: awe.data.graph.dom.Node):
        # Text fragments don't have visuals, but inherit them from their parents.
        if node.is_text:
            node = node.parent

        return [
            f
            for a in self.visual_attributes
            for f in a.compute(awe.data.visual.attribute.AttributeContext(
                node=node,
                extraction=self.extraction,
            ))
        ]
