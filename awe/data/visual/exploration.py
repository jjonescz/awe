"""
Utilities for displaying page screenshots with target nodes highlighted.
"""

import io
import itertools
import math
import os
import sys

import matplotlib.axes
import matplotlib.cm
import matplotlib.patches
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import awe.data.set.pages


def plot_websites(websites: list[awe.data.set.pages.Website], n_cols: int = 1):
    """Plots websites and their first `n_cols` pages that have screenshots."""

    return plot_pages([
        tuple(itertools.islice(
            (p for p in w.pages if os.path.exists(p.screenshot_path)),
            n_cols
        ))
        for w in websites
    ])

def plot_pages(
    pages: list[tuple[awe.data.set.pages.Page]],
    set_title: bool = True,
):
    """Plots `pages` into a grid."""

    return plot_explorers(
        [[PageExplorer(page) for page in row] for row in pages if row],
        set_title=set_title
    )

def plot_explorers(
    explorers: list[list['PageExplorer']],
    set_title: bool = True
):
    """Like `plot_pages`, but `PageExplorer`s are created manually."""

    n_cols = max(len(row) for row in explorers)

    # Find page dimensions.
    heights = [max(e.height/100 for e in row) for row in explorers]
    height = sum(heights)

    fig, axs = plt.subplots(len(explorers), n_cols,
        figsize=(10 * n_cols, height),
        facecolor='white',
        gridspec_kw={'height_ratios': heights}
    )
    axs = axs.flatten() if len(explorers) > 1 or n_cols > 1 else [axs]
    explorers = [e for row in explorers for e in row]
    for ax, e in tqdm(zip(axs, explorers), desc='pages', total=len(explorers)):
        e.plot_screenshot_with_boxes(ax)
        if set_title:
            ax.set_title(e.page.website.name)
    return fig

class PageExplorer:
    """Explores one page."""

    def __init__(self,
        page: awe.data.set.pages.Page,
        crop: bool = True,
        init_page: bool = True,
    ):
        # Load visuals.
        self.page = page
        self.page_dom = self.page.dom
        if init_page:
            self.page_labels = self.page.get_labels()
            page_visuals = self.page.load_visuals()
            self.page_dom.init_nodes()
            page_visuals.fill_tree_light(self.page_dom)
        else:
            self.page_labels = self.page.labels

        if crop:
            min_y, max_y = self._find_y_bounds()
            self.min_y = max(0, math.floor(min_y) - 5)
            self.max_y = math.ceil(max_y) + 5
        else:
            self.min_y = 0
            self.max_y = self._find_max_y()

        # Clamp pages that are too tall.
        self.max_y = min(self.min_y + 10_000, self.max_y)

    @property
    def height(self):
        return self.max_y - self.min_y

    def _find_y_bounds(self):
        """Finds ys for cropping."""

        min_y, max_y = sys.maxsize, 0
        for label_key in self.page_labels.label_keys:
            for labeled_node in self.page_labels.get_labeled_nodes(label_key):
                node = self.page_dom.find_parsed_node(labeled_node)
                if (b := node.box) is not None:
                    if b.y < min_y:
                        min_y = b.y
                    if (y := b.y + b.height) > max_y:
                        max_y = y

        return min_y, max_y

    def _find_max_y(self):
        """Finds maximum Y."""

        return max(
            math.ceil(n.box.y + n.box.height) - 1
            for n in self.page_dom.nodes
            if n.box is not None
        )

    def plot_screenshot_with_boxes(self, ax: matplotlib.axes.Axes):
        """Plots page screenshot with target node bounding boxes to `ax`."""

        # Load the screenshot.
        if self.page.screenshot_bytes is not None:
            img_source = io.BytesIO(self.page.screenshot_bytes)
        else:
            img_source = self.page.screenshot_path
        im = plt.imread(img_source)

        # Crop the screenshot.
        im = im[self.min_y:self.max_y + 1, :, :]

        # Plot the screenshot.
        ax.imshow(im)

        # Plot bounding boxes.
        cmap = matplotlib.cm.get_cmap(
            name='Set1',
            lut=len(self.page_labels.label_keys)
        )
        rects = {}
        for idx, label_key in enumerate(self.page_labels.label_keys):
            for labeled_node in self.page_labels.get_labeled_nodes(label_key):
                node = self.page_dom.find_parsed_node(labeled_node)
                if (b := node.box) is not None:
                    rect = matplotlib.patches.Rectangle(
                        xy=(b.x, b.y - self.min_y),
                        width=b.width,
                        height=b.height,
                        fill=False,
                        edgecolor=cmap(idx),
                        linewidth=2,
                        label=label_key,
                    )
                    rects[label_key] = rect
                    ax.add_patch(rect)

        # Show legend.
        ax.legend(rects.values(), rects.keys())
