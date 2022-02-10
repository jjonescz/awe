import dataclasses
from typing import TYPE_CHECKING

import torch
from torch_geometric import data

if TYPE_CHECKING:
    import awe.training.params

ModelInput = data.Batch

@dataclasses.dataclass
class ModelOutput:
    loss: torch.FloatTensor

@dataclasses.dataclass
class Prediction:
    batch: ModelInput
    outputs: ModelOutput

class Model(torch.nn.Module):
    def __init__(self,
        params: 'awe.training.params.TrainerParams',
    ):
        super().__init__()
        self.params = params

    def create_optimizer(self):
        return torch.optim.Adam(self.parameters(),
            lr=(self.lr or self.params.learning_rate)
        )

    def forward(self, batch: ModelInput) -> ModelOutput:
        pass
