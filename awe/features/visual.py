from typing import TYPE_CHECKING

import torch

import awe.data.graph.dom
import awe.data.visual.attribute
import awe.data.visual.context
import awe.features.feature

if TYPE_CHECKING:
    import awe.model.classifier


class Visuals(awe.features.feature.Feature):
    """Visual features."""

    extraction: awe.data.visual.context.Extraction

    def __post_init__(self, restoring: bool):
        self.extraction = awe.data.visual.context.Extraction()
        if restoring:
            self.freeze()

        # Filter visual attributes according to training params.
        enabled = self.trainer.params.enabled_visuals
        disabled = self.trainer.params.disabled_visuals
        self.visual_attributes = [
            a
            for a in awe.data.visual.attribute.VISUAL_ATTRIBUTES.values()
            if enabled is None or a.name in enabled
            if disabled is None or a.name not in disabled
        ]

    @property
    def labels(self):
        return [
            l
            for a in self.visual_attributes
            for l in a.get_labels()
        ]

    def prepare(self, node: awe.data.graph.dom.Node, train: bool):
        self._compute(node, freezed=not train)

    def freeze(self):
        self.extraction.freeze()

    def compute(self, batch: 'awe.model.classifier.ModelInput'):
        return torch.tensor(
            [
                self._compute(node, freezed=True)
                for node in batch
            ],
            dtype=torch.float32,
            device=self.trainer.device,
        )

    def _compute(self, node: awe.data.graph.dom.Node, freezed: bool):
        # Text fragments don't have visuals, but inherit them from their parents.
        if node.is_text:
            node = node.parent

        return [
            f
            for a in self.visual_attributes
            for f in a.select(
                awe.data.visual.attribute.AttributeContext(
                    attribute=a,
                    node=node,
                    extraction=self.extraction,
                    freezed=freezed,
                )
            )
        ]
