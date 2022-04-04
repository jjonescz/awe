import matplotlib.patches
import matplotlib.pyplot as plt

import awe.data.set.pages


def plot_screenshot_with_boxes(page: awe.data.set.pages.Page):
    # Load visuals.
    page_dom = page.dom
    page_labels = page.get_labels()
    page_visuals = page.load_visuals()
    page_dom.init_nodes()
    page_visuals.fill_tree_boxes(page_dom)
    page_dom.filter_nodes()

    # Plot the screenshot.
    im = plt.imread(page.screenshot_path)
    fig, ax = plt.subplots(figsize=(20,50))
    ax.imshow(im)

    # Plot bounding boxes.
    for label_key in page_labels.label_keys:
        for labeled_node in page_labels.get_labeled_nodes(label_key):
            node = page_dom.find_parsed_node(labeled_node)
            if (b := node.box) is not None:
                rect = matplotlib.patches.Rectangle(
                    xy=(b.x, b.y),
                    width=b.width,
                    height=b.height,
                    fill=False,
                    edgecolor='red',
                    linewidth=2
                )
                ax.add_patch(rect)

    return fig, ax
