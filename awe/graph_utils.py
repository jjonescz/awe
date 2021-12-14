import networkx as nx

from awe import awe_graph, features


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

    def link_parent_of(self, node: awe_graph.HtmlNode):
        if node.parent is not None and node.parent.dataset_index is not None:
            self.graph.add_edge(node.dataset_index, node.parent.dataset_index)
            return True
        return False

    def link_parents(self):
        for node in self.ctx.nodes:
            self.link_parent_of(node)

    def get_children(self, node: awe_graph.HtmlNode):
        return [n for n in node.children if n.dataset_index is not None]

    def link_children_of(self, node: awe_graph.HtmlNode):
        children = self.get_children(node)
        if len(children) == 0:
            return False
        for prev, next in zip(children, children[1:]):
            self.graph.add_edge(prev.dataset_index, next.dataset_index)
        return True

    def link_children(self):
        for node in self.ctx.nodes:
            self.link_children_of(node)

    def link_children_or_parents(self):
        for node in self.ctx.nodes:
            if (
                node.parent is not None and
                len(self.get_children(node.parent)) == 1
            ):
                self.link_parent_of(node)
            self.link_children_of(node)
