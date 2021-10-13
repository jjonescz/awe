from typing import Any, Callable, Tuple, Union

from torch.utils import data

from awe import awe_graph


class AweDataset(data.Dataset):
    current: Union[Tuple[int, list[awe_graph.HtmlNode]], Tuple[None, None]] = (None, None)

    def __init__(self,
        pages: list[awe_graph.HtmlPage],
        page_transform: Callable[[awe_graph.HtmlPage], list[awe_graph.HtmlNode]] = None,
        transform: Callable[[awe_graph.HtmlNode], Any] = None,
        target_transform: Callable[[list[str]], Any] = None
    ):
        self.pages = pages
        self.page_transform = page_transform or (lambda page: page.nodes)
        self.transform = transform
        self.target_transform = target_transform
        self.node_counts = [len(list(page.nodes)) for page in pages]

    def __len__(self):
        return len(self.pages)

    def _find_page(self, index: int):
        for page_index, node_count in enumerate(self.node_counts):
            if index < node_count:
                return page_index, index
            index -= node_count
        raise IndexError()

    def __getitem__(self, index: int):
        page_index, node_index = self._find_page(index)
        current_index, current_nodes = self.current
        if current_index != index:
            current_index = index
            page = self.pages[page_index]
            current_nodes = self.page_transform(page)
            self.current = current_index, current_nodes
        node = current_nodes[node_index]
        labels = node.labels
        if self.transform is not None:
            node = self.transform(node)
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        return node, labels
