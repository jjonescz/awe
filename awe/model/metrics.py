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
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def to_vector(self):
        return torch.FloatTensor([self.precision, self.recall, self.f1])

    @staticmethod
    def from_vector(vector: torch.FloatTensor):
        return F1Metrics(vector[0].item(), vector[1].item(), vector[2].item())

    @staticmethod
    def compute(stats: PredStats):
        if (stats.true_positives + stats.false_positives) == 0:
            precision = 0.0
        else:
            precision = stats.true_positives / (stats.true_positives + stats.false_positives)
        if (stats.true_positives + stats.false_negatives) == 0:
            recall = 0.0
        else:
            recall = stats.true_positives / (stats.true_positives + stats.false_negatives)
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        return F1Metrics(precision, recall, f1)

    def to_dict(self, prefix: str = ''):
        return {
            f'{prefix}precision': self.precision,
            f'{prefix}recall': self.recall,
            f'{prefix}f1': self.f1,
        }
