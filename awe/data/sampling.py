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
        train: bool = False
    ):
        """
        Pass `prepare` to run feature preparation on the pages (should be only
        done for training pages).
        """

        # Prepare DOM trees.
        for page in tqdm(pages, desc=f'init {desc}'):
            page: awe.data.set.pages.Page
            if page.cache_dom().root is None:
                page.dom.init_nodes()

                page.dom.init_labels()

        # Find variable nodes.
        if self.trainer.params.classify_only_variable_nodes:
            websites = {}
            for page in pages:
                websites.setdefault(page.website.name, page.website)
            for website in tqdm(
                websites.values(),
                desc=f'find variable nodes in {desc}'
            ):
                website: awe.data.set.pages.Website
                website.find_variable_nodes()

        # Compute friend cycles.
        for page in tqdm(pages, desc=f'prepare {desc}'):
            if (
                self.trainer.params.friend_cycles
                and not page.dom.friend_cycles_computed
            ):
                page.dom.compute_friend_cycles(
                    max_friends=self.trainer.params.max_friends,
                    only_variable_nodes=self.trainer.params.classify_only_variable_nodes
                )

            if train:
                # Add all label keys to a map.
                for label_key in page.labels.label_keys:
                    self.trainer.label_map.map_label_to_id(label_key)

            # Prepare features.
            self.trainer.extractor.prepare_page(page.dom, train=train)

            # Initialize features.
            if train:
                self.trainer.extractor.initialize()

        # Select nodes.
        return [
            node
            for page in pages
            for node in self.select_nodes_for_page(page)
        ]

    def select_nodes_for_page(self,
        page: awe.data.set.pages.Page
    ) -> list[Sample]:
        if self.trainer.params.classify_only_variable_nodes:
            return [n for n in page.dom.nodes if n.is_variable_text]
        if self.trainer.params.classify_only_text_nodes:
            return [n for n in page.dom.nodes if n.is_text]
        return page.dom.nodes

class Collater:
    """
    Prepares data for model. When called, takes batch of samples and returns
    batch of model inputs.
    """

    def __call__(self, samples: list[Sample]):
        return samples
