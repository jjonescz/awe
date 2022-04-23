from typing import TYPE_CHECKING, Optional, TypeVar

import awe.data.glove
import awe.data.graph.dom
import awe.data.set.pages
import awe.features.dom
import awe.features.feature
import awe.features.text
import awe.features.visual
import awe.training.params
import awe.utils

if TYPE_CHECKING:
    import awe.training.trainer

T = TypeVar('T')

class Extractor:
    """Extracts features."""

    features: list[awe.features.feature.Feature]

    def __init__(self, trainer: 'awe.training.trainer.Trainer'):
        self.trainer = trainer
        self.features = []
        if (self.trainer.params.word_vector_function is not None or
            self.trainer.params.tokenize_node_attrs):
            self.features.append(awe.features.text.WordIdentifiers(trainer))
        if self.trainer.params.tag_name_embedding:
            self.features.append(awe.features.dom.HtmlTag(trainer))
        if self.trainer.params.position:
            self.features.append(awe.features.dom.Position(trainer))
        if self.trainer.params.load_visuals:
            self.features.append(awe.features.visual.Visuals(trainer))

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

    def freeze(self):
        for feature in self.features:
            feature.freeze()

    def enable_cache(self, enable: bool = True):
        for feature in self.features:
            feature.enable_cache(enable=enable)

    def restore_features(self, features: list[awe.features.feature.Feature]):
        self.features = features
        for feature in features:
            feature.trainer = self.trainer
            feature.__post_init__(restoring=True)
