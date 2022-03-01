import abc
from typing import TYPE_CHECKING

import torch

import awe.data.graph.dom
import awe.features.context

if TYPE_CHECKING:
    import awe.model.classifier
    import awe.training.trainer


class Feature(abc.ABC):
    def __init__(self, trainer: 'awe.training.trainer.Trainer'):
        self.trainer = trainer
        self.__post_init__()

    def __post_init__(self):
        """Can be used by derived classes to do initialization."""

    def prepare(self, node: awe.data.graph.dom.Node, train: bool):
        """
        Prepares this feature for the given `node`.

        This method runs for all nodes before initializing and computing the
        features. Can be used for example to populate a global word dictionary.
        """

    def initialize(self):
        """Work needed to be done so that this feature can be computed."""

    @abc.abstractmethod
    def compute(self, batch: 'awe.model.classifier.ModelInput') -> torch.FloatTensor:
        """Computes a feature vector for the given `batch` of nodes."""
