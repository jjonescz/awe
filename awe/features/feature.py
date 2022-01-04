from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

import torch

# pylint: disable=wildcard-import, unused-wildcard-import
from awe.features.context import *

if TYPE_CHECKING:
    from awe import awe_graph

T = TypeVar('T')

class Feature(ABC):
    @property
    @abstractmethod
    def summary(self):
        """User-friendly details."""

    def initialize(self, context: LiveContext):
        """Work needed to be done so that this feature can be computed."""

    def prepare(self, node: 'awe_graph.HtmlNode', context: RootContext):
        """
        Prepares feature for the given `node`.

        This method runs for all nodes before initializing and computing the
        features. Can be used for example to populate a global word dictionary.
        """

    @abstractmethod
    def compute(self,
        node: 'awe_graph.HtmlNode',
        context: PageContext) -> torch.FloatTensor:
        """
        Computes feature vector for the given `node`.

        This vector will be serialized.
        """

class DirectFeature(Feature):
    """
    These features are appended to `torch_geometric.data.Data.x` without further
    processing.
    """

    @property
    @abstractmethod
    def labels(self) -> list[str]:
        """Column names of the resulting feature vector."""

    @property
    def summary(self):
        return { 'labels': self.labels }

class IndirectFeature(Feature):
    """
    These features are appended to `torch_geometric.data.Data` by their `label`.
    """

    @property
    @abstractmethod
    def label(self) -> str:
        """Attribute name on `Data` for the resulting feature vector."""

    @property
    def summary(self):
        return { 'label': self.label }

    # pylint: disable-next=unused-argument,no-self-use
    def update(self, context: RootContext, vector: torch.FloatTensor):
        """
        Updates (usually shape of) `vector` to match data in `RootContext`.

        This can be used to consolidate features computed earlier with features
        computed for new data which can be padded to longer size.
        """
        return vector
