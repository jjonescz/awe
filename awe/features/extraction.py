from typing import TYPE_CHECKING, Optional, TypeVar

import awe.data.glove
import awe.data.graph.dom
import awe.data.set.pages
import awe.features.dom
import awe.features.text
import awe.training.params
import awe.utils

if TYPE_CHECKING:
    import awe.training.trainer

T = TypeVar('T')

class Extractor:
    """Extracts features."""

    def __init__(self, trainer: 'awe.training.trainer.Trainer'):
        self.trainer = trainer
        self.features = []
        if self.trainer.params.word_vector_function is not None:
            self.features.append(awe.features.text.WordIdentifiers(trainer))

    def prepare_page(self, page_dom: awe.data.graph.dom.Dom, train: bool):
        """Prepares features for the `page`."""

        for feature in self.features:
            for node in page_dom.nodes:
                feature.prepare(node=node, train=train)

    def get_feature(self, cls: type[T]) -> Optional[T]:
        for feature in self.features:
            if awe.utils.same_types(feature.__class__, cls):
                return feature
        return None

    def has_feature(self, cls: type):
        return self.get_feature(cls) is not None

    def initialize(self):
        for feature in self.features:
            feature.initialize()
