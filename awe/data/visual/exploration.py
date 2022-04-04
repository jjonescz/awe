import math

import matplotlib.cm
import matplotlib.patches
import matplotlib.pyplot as plt

import awe.data.set.pages


def plot_screenshot_with_boxes(
    ax: matplotlib.axes.Axes,
    page: awe.data.set.pages.Page
):
    # Load visuals.
    page_dom = page.dom
    page_labels = page.get_labels()
    page_visuals = page.load_visuals()
    page_dom.init_nodes()
    page_visuals.fill_tree_boxes(page_dom)

    # Find max y.
    max_y = 0
    for label_key in page_labels.label_keys:
        for labeled_node in page_labels.get_labeled_nodes(label_key):
            node = page_dom.find_parsed_node(labeled_node)
            if (b := node.box) is not None:
                if (y := b.y + b.height) > max_y:
                    max_y = y

    # Crop the screenshot.
    im = plt.imread(page.screenshot_path)
    im = im[:math.ceil(max_y) + 5, :, :]

    # Plot the screenshot.
    ax.imshow(im)

    # Plot bounding boxes.
    cmap = matplotlib.cm.get_cmap(
        name='Set1',
        lut=len(page_labels.label_keys)
    )
    rects = {}
    for idx, label_key in enumerate(page_labels.label_keys):
        for labeled_node in page_labels.get_labeled_nodes(label_key):
            node = page_dom.find_parsed_node(labeled_node)
            if (b := node.box) is not None:
                rect = matplotlib.patches.Rectangle(
                    xy=(b.x, b.y),
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
