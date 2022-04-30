import collections
import hashlib
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch.utils.data
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

class Sampler:
    """Prepares data samples for model."""

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

    def prepare(self):
        params = self.trainer.params

        if params.classify_only_variable_nodes:
            self.find_variable_nodes()

    def load(self):
        """Prepares all pages and returns list of sampled nodes."""

        params = self.trainer.params

        self.prepare()

        self.init_dom_trees()

        self.prepare_features()

        pages = self.pages
        if params.validate_data:
            if not self.validate(self.pages, progress_bar=True):
                if params.ignore_invalid_pages:
                    pages = [p for p in pages if p.valid]
                    warnings.warn(f'Ignored invalid pages in {self.desc!r} ' +
                        f'({len(self.pages)} -> {len(pages)}).')
                else:
                    raise RuntimeError(f'Validation failed for {self.desc!r}.')

        return [n for p in pages for n in p.dom.nodes if n.sample]

    def load_one(self, page: awe.data.set.pages.Page):
        """
        Prepares one page (lazily) and returns list of its nodes.

        Should be used via `LazySampler`.
        """

        params = self.trainer.params

        try:
            if page.cache_dom().root is None:
                self.init_dom_tree(page)
            self.prepare_features_for_page(page)

            if params.validate_data:
                if not self.validate([page], progress_bar=False):
                    if params.ignore_invalid_pages:
                        return []
                    raise RuntimeError(f'Validation failed for {self.desc!r}.')

            return [n for n in page.dom.nodes if n.sample]
        finally:
            page.clear_dom()

    def find_variable_nodes(self):
        """Finds variable nodes for all websites that will be sampled from."""

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
        """Parses HTML into DOM trees of all pages."""

        for page in tqdm(self.pages, desc=f'init {self.desc}'):
            page: awe.data.set.pages.Page
            if page.cache_dom().root is None:
                self.init_dom_tree(page)

    def prepare_features(self):
        """Prepares features for all pages."""

        for page in tqdm(self.pages, desc=f'prepare {self.desc}'):
            self.prepare_features_for_page(page)

        # Freeze features.
        if self.train:
            self.trainer.extractor.freeze()

    def validate(self, pages: list[awe.data.set.pages.Page], progress_bar: bool):
        """Validates `pages`."""

        validator = awe.data.validation.Validator(
            only_cached_dom=True,
            # It is not needed to validate visuals as that's automatically
            # performed when they are loaded (a few lines above).
            visuals=False
        )
        validator.validate_pages(pages,
            progress_bar=f'validate {self.desc}' if progress_bar else None
        )
        return validator.num_invalid == 0

    def is_variable_node(self, node: awe.data.graph.dom.Node):
        """Determines whether `node` is in the set of variable nodes."""

        return node.get_xpath() in self.var_nodes[node.dom.page.website.name]

    def init_dom_tree(self, page: awe.data.set.pages.Page):
        """Parses HTML into DOM tree for one `page`."""

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

        check_sampled_nodes(page)

    def prepare_features_for_page(self, page: awe.data.set.pages.Page):
        """Prepares features for one `page`."""

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
        """Loads visuals into DOM tree of one `page`."""

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
        """Loads labels into DOM tree of one `page`."""

        params = self.trainer.params

        page.dom.init_labels(
            propagate_to_leaves=params.propagate_labels_to_leaves
        )

    def should_sample(self, node: awe.data.graph.dom.Node):
        """Decides whether `node` should be sampled."""

        params = self.trainer.params

        if params.classify_only_variable_nodes:
            return (
                self.is_text_or_correct_leaf(node) and
                (not self.train or self.is_variable_node(node)) and
                not self.should_cutoff(node)
            )

        if params.classify_only_text_nodes:
            return (
                self.is_text_or_correct_leaf(node) and
                not self.should_cutoff(node)
            )

        return True

    def should_cutoff(self, node: awe.data.graph.dom.Node):
        """Applies `Params.none_cutoff`."""

        params = self.trainer.params

        # If the node is not labeled, cut it off with some probability (to
        # have more balanced data).
        return (
            self.train and
            not node.label_keys and
            params.none_cutoff is not None and
            self.rng.integers(0, 100_000) >= params.none_cutoff
        )

    def is_text_or_correct_leaf(self, node: awe.data.graph.dom.Node):
        """Determines whether `node` should be sampled according to `Params`."""

        params = self.trainer.params

        return (
            node.is_text or
            (node.is_leaf and node.html_tag in
            params.classify_also_html_tags)
        )

def check_sampled_nodes(page: awe.data.set.pages.Page):
    """Checks that all target nodes are indeed sampled."""

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

class LazySampler(torch.utils.data.IterableDataset):
    """`IterableDataset` implementation for `Sampler`."""

    def __init__(self, sampler: Sampler):
        super().__init__()
        self.sampler = sampler
        self.sampler.prepare()
        self.num_nodes = 0
        self.num_pages = 0

    def __len__(self):
        # Estimate length of not-yet-prepared pages by using the average number
        # of nodes per page of prepared pages.
        avg_nodes = (
            self.num_nodes / self.num_pages
            if self.num_pages != 0 else 2_000
        )
        remaining_pages = len(self.sampler.pages) - self.num_pages
        remaining_nodes = math.floor(avg_nodes * remaining_pages)
        return self.num_nodes + remaining_nodes

    def __getitem__(self, index):
        raise NotImplementedError()

    def __iter__(self):
        for p in self.sampler.pages:
            nodes = self.sampler.load_one(p)
            self.num_nodes += len(nodes)
            self.num_pages += 1
            for n in nodes:
                yield n

class Collater:
    """
    Prepares data for model. When called, takes batch of samples and returns
    batch of model inputs.
    """

    def __call__(self, samples: list[awe.data.graph.dom.Node]):
        # Note that even this simple function is needed because PyTorch would
        # complain that the input is not a Tensor if using the default collater.
        return samples
