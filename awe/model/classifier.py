import dataclasses
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn

import awe.data.graph.dom
import awe.features.dom
import awe.features.extraction

if TYPE_CHECKING:
    import awe.training.trainer

ModelInput = list[awe.data.graph.dom.Node]

@dataclasses.dataclass
class ModelOutput:
    loss: torch.FloatTensor

@dataclasses.dataclass
class Prediction:
    batch: ModelInput
    outputs: ModelOutput

class Model(torch.nn.Module):
    def __init__(self,
        trainer: 'awe.training.trainer.Trainer',
        lr: Optional[float] = None,
    ):
        super().__init__()
        self.trainer = trainer
        self.lr = lr

        input_features = 0

        # HTML tag name embedding
        num_html_tags = len(self.trainer.extractor.context.html_tags)
        if num_html_tags > 0:
            embedding_dim = 32
            self.tag_embedding = torch.nn.Embedding(
                num_html_tags,
                embedding_dim
            )
            input_features += embedding_dim

        # Classification head
        D = 64
        num_labels = len(self.trainer.label_map.id_to_label) + 1
        self.head = torch.nn.Sequential(
            torch.nn.Linear(input_features, 2 * D),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(2 * D, D),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(D, num_labels)
        )

        self.loss = torch.nn.CrossEntropyLoss(
            #weight=torch.FloatTensor(params.label_weights),
            #label_smoothing=params.label_smoothing
        )

    def create_optimizer(self):
        return torch.optim.Adam(self.parameters(),
            lr=(self.lr or self.trainer.params.learning_rate)
        )

    def forward(self, batch: ModelInput) -> ModelOutput:
        # Embed HTML tag names.
        html_tag = self.trainer.extractor.get_feature(awe.features.dom.HtmlTag)
        if html_tag is not None:
            tag_ids = torch.IntTensor(
                [
                    html_tag.compute(node)
                    for node in batch
                ],
                device=self.device
            )
            x = self.tag_embedding(tag_ids)

        # Classify features.
        x = self.head(x)

        # Find out gold labels.
        gold_labels = [
            self.trainer.label_map.get_label_id(node)
            for node in batch
        ]
        loss = self.loss(x, gold_labels)

        return ModelOutput(loss=loss)
