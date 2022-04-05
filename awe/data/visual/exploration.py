import math
import sys

import matplotlib.cm
import matplotlib.patches
import matplotlib.pyplot as plt

import awe.data.set.pages


class PageExplorer:
    def __init__(self, page: awe.data.set.pages.Page):
        # Load visuals.
        self.page = page
        self.page_dom = self.page.dom
        self.page_labels = self.page.get_labels()
        self.page_visuals = self.page.load_visuals()
        self.page_dom.init_nodes()
        self.page_visuals.fill_tree_light(self.page_dom)

    def find_y_bounds(self):
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

    def plot_screenshot_with_boxes(self, ax: matplotlib.axes.Axes):
        # Crop the screenshot.
        im = plt.imread(self.page.screenshot_path)
        min_y, max_y = self.find_y_bounds()
        offset = math.floor(min_y) - 5
        im = im[offset:math.ceil(max_y) + 5, :, :]

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
                        xy=(b.x, b.y - offset),
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
