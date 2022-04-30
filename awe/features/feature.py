import abc
from typing import TYPE_CHECKING

import torch

import awe.data.graph.dom
import awe.utils

if TYPE_CHECKING:
    import awe.training.trainer


class Feature(awe.utils.PickleSubset, abc.ABC):
    """Interface for all features."""

    def __init__(self, trainer: 'awe.training.trainer.Trainer'):
        self.trainer = trainer
        self.__post_init__(restoring=False)

    def get_pickled_keys(self):
        return ()

    def __post_init__(self, restoring: bool):
        """Can be used by derived classes to do initialization."""

    def prepare(self, node: awe.data.graph.dom.Node, train: bool):
        """
        Prepares this feature for the given `node`.

        This method runs for all nodes before initializing and computing the
        features. Can be used for example to populate a global word dictionary.
        """

    def enable_cache(self, enable: bool = True):
        """Enables or disables (and clears) cache."""

    def freeze(self):
        """
        Called after preparing the feature for all training data. Should make
        the feature pickleable.
        """

    @abc.abstractmethod
    def compute(self, batch: list[awe.data.graph.dom.Node]) -> torch.FloatTensor:
        """Computes a feature vector for the given `batch` of nodes."""
