from typing import TYPE_CHECKING, Callable

from tqdm.auto import tqdm

import awe.data.graph.dom
import awe.data.parsing
import awe.data.set.pages
import awe.data.validation
import awe.features.extraction
import awe.training.params

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
                if not self.trainer.params.load_visuals:
                    # If not loading visuals, we can filter nodes in one pass.
                    awe.data.parsing.filter_tree(page.dom.tree)
                page.dom.init_nodes()
                if self.trainer.params.load_visuals:
                    # Load visuals.
                    page_visuals = page.load_visuals()
                    page_visuals.fill_tree_boxes(page.dom)

                    page.dom.filter_nodes()
                page.dom.init_labels(
                    propagate_to_leaves=
                        self.trainer.params.propagate_labels_to_leaves
                )

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

        for page in tqdm(pages, desc=f'prepare {desc}'):
            # Compute friend cycles.
            if (
                self.trainer.params.friend_cycles
                and not page.dom.friend_cycles_computed
            ):
                page.dom.compute_friend_cycles(
                    max_friends=self.trainer.params.max_friends,
                    only_variable_nodes=self.trainer.params.classify_only_variable_nodes
                )

            # Compute visual neighbors.
            if (
                self.trainer.params.visual_neighbors
                and not page.dom.visual_neighbors_computed
            ):
                neighbor_distance = self.trainer.params.neighbor_distance
                D = awe.training.params.VisualNeighborDistance
                if neighbor_distance == D.center_point:
                    f = page.dom.compute_visual_neighbors
                elif neighbor_distance == D.rect:
                    f = page.dom.compute_visual_neighbors_rect
                else:
                    raise ValueError(
                        f'Unrecognized param {neighbor_distance=}.')
                f(n_neighbors=self.trainer.params.n_neighbors)

            if train:
                # Add all label keys to a map.
                for label_key in page.labels.label_keys:
                    self.trainer.label_map.map_label_to_id(label_key)

            # Prepare features.
            self.trainer.extractor.prepare_page(page.dom, train=train)

            # Initialize features.
            if train:
                self.trainer.extractor.initialize()

        # Validate.
        if self.trainer.params.validate_data:
            validator = awe.data.validation.Validator(
                # It is not needed to validate visuals as that's automatically
                # performed when they are loaded (a few lines above).
                visuals=False
            )
            validator.validate_pages(pages, progress_bar=f'validate {desc}')

        # Select nodes.
        return [
            node
            for page in pages
            for node in self.select_nodes_for_page(page)
        ]

    def select_nodes_for_page(self,
        page: awe.data.set.pages.Page
    ) -> list[Sample]:
        if self.trainer.params.visual_neighbors:
            return filter_nodes(page, lambda n: n.visual_neighbors is not None)
        if self.trainer.params.classify_only_variable_nodes:
            return filter_nodes(page, lambda n: n.is_variable_text)
        if self.trainer.params.classify_only_text_nodes:
            return filter_nodes(page, lambda n: n.is_text)
        return page.dom.nodes

def filter_nodes(
    page: awe.data.set.pages.Page,
    predicate: Callable[[awe.data.graph.dom.Node], bool]
):
    for node in page.dom.nodes:
        if predicate(node):
            yield node
        # Check that excluded nodes are not labeled.
        elif len(node.label_keys) != 0:
            raise RuntimeError(f'Excluded node {node.get_xpath()!r} ' +
                f'labeled {node.label_keys!r} ({page.html_path!r}).')

class Collater:
    """
    Prepares data for model. When called, takes batch of samples and returns
    batch of model inputs.
    """

    def __call__(self, samples: list[Sample]):
        return samples
