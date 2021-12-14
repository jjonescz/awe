import networkx as nx

from awe import features


def to_networkx(ctx: features.PageContextBase):
    graph = nx.DiGraph()

    for node in ctx.nodes:
        graph.add_node(
            node.dataset_index,
            id=node.element.get('id') or '' if not node.is_text else '',
            tag=node.tag_name,
            text=node.text if node.is_text else '',
            label=node.labels
        )

    for node in ctx.nodes:
        if node.parent is not None:
            graph.add_edge(node.dataset_index, node.parent.dataset_index)

    return graph
