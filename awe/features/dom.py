from typing import TYPE_CHECKING

import awe.features.feature
import awe.data.graph.dom

if TYPE_CHECKING:
    import awe.features.extraction


class HtmlTag(awe.features.feature.Feature):

    def prepare(self,
        node: awe.data.graph.dom.Node,
        extractor: 'awe.features.extraction.Extractor'
    ):
        extractor.context.html_tags.add(node.html_tag)

    def initialize(self, extractor: 'awe.features.extraction.Extractor'):
        # Map all found HTML tags to numbers. Note that 0 is reserved for
        # "unknown" tags.
        extractor.context.html_tag_ids = {
            c: i + 1
            for i, c in enumerate(extractor.context.html_tags)
        }

    def compute(self,
        node: awe.data.graph.dom.Node,
        extractor: 'awe.features.extraction.Extractor'
    ):
        return extractor.context.html_tag_ids.get(node.html_tag, 0)
