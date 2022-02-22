import abc
from typing import TYPE_CHECKING

import torch

import awe.data.graph.dom
import awe.features.context

if TYPE_CHECKING:
    import awe.features.extraction


class Feature(abc.ABC):
    def prepare(self,
        node: awe.data.graph.dom.Node,
        extractor: 'awe.features.extraction.Extractor'
    ):
        """
        Prepares this feature for the given `node`.

        This method runs for all nodes before initializing and computing the
        features. Can be used for example to populate a global word dictionary.
        """

    @abc.abstractmethod
    def initialize(self, extractor: 'awe.features.extraction.Extractor') -> int:
        """
        Work needed to be done so that this feature can be computed.

        Returns length of the feature vector that will be `compute`d.
        """

    @abc.abstractmethod
    def compute(self,
        node: awe.data.graph.dom.Node,
        extractor: 'awe.features.extraction.Extractor'
    ) -> torch.FloatTensor:
        """Computes a feature vector for the given `node`."""
