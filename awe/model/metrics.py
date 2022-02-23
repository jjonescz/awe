import dataclasses

import torch


@dataclasses.dataclass
class PredStats:
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

@dataclasses.dataclass
class F1Metrics:
    precision: float
    recall: float
    f1: float

    def to_vector(self):
        return torch.FloatTensor([self.precision, self.recall, self.f1])

    @staticmethod
    def from_vector(vector: torch.FloatTensor):
        return F1Metrics(vector[0].item(), vector[1].item(), vector[2].item())

    @staticmethod
    def compute(*,
        true_positives: int,
        false_positives: int,
        false_negatives: int
    ):
        if (true_positives + false_positives) == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)
        if (true_positives + false_negatives) == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        return F1Metrics(precision, recall, f1)
