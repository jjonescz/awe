from typing import Optional, TypeVar

import awe.data.glove
import awe.data.graph.dom
import awe.data.set.pages
import awe.features.context
import awe.features.dom
import awe.features.text
import awe.training.params

T = TypeVar('T')

class Extractor:
    """Extracts features."""

    def __init__(self, params: awe.training.params.Params):
        self.params = params
        self.context = awe.features.context.RootContext()
        self.features = [
            awe.features.dom.HtmlTag(self),
            #awe.features.text.WordIdentifiers(self),
        ]

    def prepare_page(self, page_dom: awe.data.graph.dom.Dom):
        """Prepares features for the `page`."""

        for feature in self.features:
            for node in page_dom.nodes:
                feature.prepare(node)

    def get_feature(self, cls: type[T]) -> Optional[T]:
        for feature in self.features:
            if isinstance(feature, cls):
                return feature
        return None

    def has_feature(self, cls: type):
        return self.get_feature(cls) is not None

    def initialize(self):
        for feature in self.features:
            feature.initialize()
