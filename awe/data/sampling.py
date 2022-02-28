from typing import TYPE_CHECKING

from tqdm.auto import tqdm

import awe.data.graph.dom
import awe.data.set.pages
import awe.features.extraction

if TYPE_CHECKING:
    import awe.training.trainer

Sample = awe.data.graph.dom.Node

class Sampler:
    """
    Prepares data samples for training. Method `load` takes pages and returns
    list of samples.
    """

    def __init__(self, trainer: 'awe.training.trainer.Trainer'):
        self.trainer = trainer

    def load(self,
        pages: list[awe.data.set.pages.Page],
        desc: str,
        prepare: bool = False
    ):
        """
        Pass `prepare` to run feature preparation on the pages (should be only
        done for training pages).
        """

        result = [
            node
            for page in tqdm(pages, desc=f'loading {desc}')
            for node in self.get_nodes_for_page(page, prepare=prepare)
        ]
        if prepare:
            self.trainer.extractor.initialize()
        return result

    def get_nodes_for_page(self,
        page: awe.data.set.pages.Page,
        prepare: bool = False
    ) -> list[Sample]:
        if page.cache_dom().root is None:
            page.dom.init_nodes()

            page.dom.init_labels()

            if self.trainer.params.friend_cycles:
                page.dom.compute_friend_cycles(
                    max_friends=self.trainer.params.max_friends
                )

        if prepare:
            for label_key in page.labels.label_keys:
                self.trainer.label_map.map_label_to_id(label_key)

            self.trainer.extractor.prepare_page(page.dom)

        return page.dom.nodes

class Collater:
    """
    Prepares data for model. When called, takes batch of samples and returns
    batch of model inputs.
    """

    def __call__(self, samples: list[Sample]):
        return samples
