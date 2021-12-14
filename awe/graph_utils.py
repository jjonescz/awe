import networkx as nx

from awe import features


def remove_single_nodes(ctx: features.PageContextBase):
    """Removes nodes with single child while preserving edges."""
    changes = 0
    changed = True
    while changed and changes < 100:
        changes += 1
        changed = False
        to_remove = []
        for node in ctx.nodes:
            if node.parent is not None and len(node.children) == 1:
                # Rewire edges.
                child = node.children[0]
                node.parent.children[node.index] = child
                child.parent = node.parent

                # Detach this node.
                to_remove.append(node)

                changed = True

        # Remove detached nodes.
        for node in to_remove:
            ctx.nodes.remove(node)

    return changes

class PageGraph:
    def __init__(self, ctx: features.PageContextBase):
        self.ctx = ctx
        self.graph = nx.DiGraph()

        for node in ctx.nodes:
            self.graph.add_node(
                node.dataset_index,
                id=node.element.get('id') or '' if not node.is_text else '',
                tag=node.tag_name,
                text=node.text if node.is_text else '',
                leaf=node.is_text,
                summary=node.summary,
                label=node.labels,
                display_size=5 * (len(node.labels) + 1),
                coords=(node.box.x, node.box.y) if node.box is not None else (0, 0),
                center=node.box.center_point if node.box is not None else (0, 0)
            )

    def link_parents(self):
        for node in self.ctx.nodes:
            if node.parent is not None and node.parent.dataset_index is not None:
                self.graph.add_edge(node.dataset_index, node.parent.dataset_index)

    def link_siblings(self):
        for node in self.ctx.nodes:
            siblings = [n for n in node.children if n.dataset_index is not None]
            for prev, next in zip(siblings, siblings[1:]):
                self.graph.add_edge(prev.dataset_index, next.dataset_index)
