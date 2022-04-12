import collections
import hashlib
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

import awe.data.graph.dom
import awe.data.parsing
import awe.data.set.pages
import awe.data.validation
import awe.features.extraction
import awe.features.visual
import awe.training.params

if TYPE_CHECKING:
    import awe.training.trainer

Sample = awe.data.graph.dom.Node

class Sampler:
    """
    Prepares data samples for training. Method `load` takes pages and returns
    list of samples.
    """

    var_nodes: dict[str, set[str]]

    def __init__(self,
        trainer: 'awe.training.trainer.Trainer',
        pages: list[awe.data.set.pages.Page],
        desc: str,
        train: bool = False
    ):
        self.trainer = trainer
        self.pages = pages
        self.desc = desc
        self.train = train

        # Compute deterministic hash of inputs.
        input_hash = hashlib.sha256()
        input_hash.update(f'{len(pages)}-{desc}'.encode('utf-8'))
        seed = np.frombuffer(input_hash.digest(), dtype='uint32')

        self.rng = np.random.default_rng(seed)
        self.visual_feat = self.trainer.extractor.get_feature(
            awe.features.visual.Visuals
        )

    def load(self):
        params = self.trainer.params

        if params.classify_only_variable_nodes:
            self.find_variable_nodes()

        self.init_dom_trees()

        self.prepare_features()

        if params.validate_data:
            self.validate()

        return [n for p in self.pages for n in p.dom.nodes if n.sample]

    def find_variable_nodes(self):
        # Find all websites contained in `pages`.
        websites = {}
        for page in self.pages:
            websites.setdefault(page.website.name, page.website)

        # For each website, get list of variable nodes.
        self.var_nodes = {
            website.name: website.find_variable_xpaths()
            for website in tqdm(
                websites.values(),
                desc=f'find variable nodes in {self.desc}'
            )
        }

    def init_dom_trees(self):
        for page in tqdm(self.pages, desc=f'init {self.desc}'):
            page: awe.data.set.pages.Page
            if page.cache_dom().root is None:
                self.init_dom_tree(page)

    def prepare_features(self):
        for page in tqdm(self.pages, desc=f'prepare {self.desc}'):
            self.prepare_features_for_page(page)

        # Freeze features.
        if self.train:
            self.trainer.extractor.freeze()

    def validate(self):
        validator = awe.data.validation.Validator(
            only_cached_dom=True,
            # It is not needed to validate visuals as that's automatically
            # performed when they are loaded (a few lines above).
            visuals=False
        )
        validator.validate_pages(self.pages,
            progress_bar=f'validate {self.desc}'
        )
        if validator.num_invalid > 0:
            raise RuntimeError(f'Validation failed for {self.desc!r}.')

    def is_variable_node(self, node: awe.data.graph.dom.Node):
        return node.get_xpath() in self.var_nodes[node.dom.page.website.name]

    def init_dom_tree(self, page: awe.data.set.pages.Page):
        params = self.trainer.params

        page.dom.init_nodes(
            # If not loading visuals, we can filter nodes in one pass.
            filter_tree=not params.load_visuals
        )
        self.init_labels(page)

        # Select nodes for classification.
        for node in page.dom.nodes:
            node.sample = self.should_sample(node)

        if params.load_visuals:
            self.load_visuals(page)
            page.dom.filter_nodes()

            # Re-compute labels (some nodes might have been filtered out).
            self.init_labels(page)

            # Deselect nodes that have no bounding box.
            if params.visual_neighbors:
                for node in page.dom.nodes:
                    if node.sample and node.box is None:
                        node.sample = False

        self.check_sampled_nodes(page)

    def prepare_features_for_page(self, page: awe.data.set.pages.Page):
        params = self.trainer.params

        # Compute friend cycles.
        if params.friend_cycles and not page.dom.friend_cycles_computed:
            page.dom.compute_friend_cycles(
                max_friends=params.max_friends,
            )

        # Compute visual neighbors.
        if params.visual_neighbors and not page.dom.visual_neighbors_computed:
            neighbor_distance = params.neighbor_distance
            D = awe.training.params.VisualNeighborDistance
            if neighbor_distance == D.center_point:
                f = page.dom.compute_visual_neighbors
            elif neighbor_distance == D.rect:
                f = page.dom.compute_visual_neighbors_rect
            else:
                raise ValueError(
                    f'Unrecognized param {neighbor_distance=}.')
            f(n_neighbors=params.n_neighbors)

        if self.train:
            # Add all label keys to a map.
            for label_key in page.labels.label_keys:
                self.trainer.label_map.map_label_to_id(label_key)

        # Prepare features.
        self.trainer.extractor.prepare_page(page.dom, train=self.train)

    def load_visuals(self, page: awe.data.set.pages.Page):
        # Mark nodes that need visuals parsed.
        for node in page.dom.nodes:
            if node.sample:
                if node.is_text:
                    node.parent.needs_visuals = True
                else:
                    node.needs_visuals = True

        # Load visuals.
        page_visuals = page.load_visuals()
        page_visuals.fill_tree_light(page.dom,
            attrs=self.visual_feat.visual_attributes \
                if self.visual_feat is not None else ()
        )

    def init_labels(self, page: awe.data.set.pages.Page):
        params = self.trainer.params

        page.dom.init_labels(
            propagate_to_leaves=params.propagate_labels_to_leaves
        )

    def should_sample(self, node: awe.data.graph.dom.Node):
        params = self.trainer.params

        # If the node is not labeled, cut it off with some probability (to
        # have more balanced data).
        if not node.label_keys and params.none_cutoff is not None:
            if self.rng.integers(0, 100_000) >= params.none_cutoff:
                return False

        if params.classify_only_variable_nodes:
            return (
                self.is_text_or_correct_leaf(node) and
                self.is_variable_node(node)
            )

        if params.classify_only_text_nodes:
            return self.is_text_or_correct_leaf(node)

        return True

    def is_text_or_correct_leaf(self, node: awe.data.graph.dom.Node):
        params = self.trainer.params

        return (
            node.is_text or
            (node.is_leaf and node.html_tag in
            params.classify_also_html_tags)
        )

    def check_sampled_nodes(self, page: awe.data.set.pages.Page):
        included = collections.defaultdict(int)
        excluded = collections.defaultdict(int)
        for node in page.dom.nodes:
            if node.sample:
                if len(node.label_keys) != 0:
                    for label_key in node.label_keys:
                        included[label_key] += 1
                yield node
            elif len(node.label_keys) != 0:
                for label_key in node.label_keys:
                    excluded[label_key] += 1

        for label_key in page.labels.label_keys:
            e = excluded.get(label_key, 0)
            if e > 0 and included.get(label_key, 0) == 0:
                raise RuntimeError(f'Excluded all {e} node(s) ' +
                    f'labeled {label_key!r} ({page.html_path!r}).')

class Collater:
    """
    Prepares data for model. When called, takes batch of samples and returns
    batch of model inputs.
    """

    def __call__(self, samples: list[Sample]):
        # Note that even this simple function is needed because PyTorch would
        # complain that the input is not a Tensor if using the default collater.
        return samples
