import dataclasses
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn

import awe.data.graph.dom
import awe.features.dom
import awe.features.extraction
import awe.data.glove
import awe.model.word_lstm

if TYPE_CHECKING:
    import awe.training.trainer

ModelInput = list[awe.data.graph.dom.Node]

@dataclasses.dataclass
class ModelOutput:
    loss: torch.FloatTensor
    logits: torch.FloatTensor
    gold_labels: torch.FloatTensor

    def get_pred_labels(self):
        return torch.argmax(self.logits, dim=-1)

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
        self.html_tag = self.trainer.extractor.get_feature(awe.features.dom.HtmlTag)
        if self.html_tag is not None:
            num_html_tags = len(self.trainer.extractor.context.html_tags) + 1
            embedding_dim = 32
            self.tag_embedding = torch.nn.Embedding(
                num_html_tags,
                embedding_dim
            )
            input_features += embedding_dim

        # Word LSTM
        if self.trainer.params.use_lstm:
            self.lstm = awe.model.word_lstm.WordLstm(self)
            out_dim = self.lstm.out_dim
            if self.trainer.params.friend_cycles:
                out_dim *= 3
            input_features += out_dim

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
        if self.html_tag is not None:
            tag_ids = self.html_tag.compute(batch) # [N]
            x = self.tag_embedding(tag_ids) # [N, embedding_dim]

        if self.trainer.params.use_lstm:
            if self.trainer.params.friend_cycles:
                # Expand partner and friend nodes.
                friend_batch = [None] * (len(batch) * 3)
                for i, n in zip(range(0, len(friend_batch), 3), batch):
                    friend_batch[i] = [n]
                    friend_batch[i + 1] = n.get_partner_set()
                    friend_batch[i + 2] = n.friends
                expanded_batch = friend_batch
            else:
                expanded_batch = [batch]

            x = self.lstm(expanded_batch) # [3N, lstm_dim]

            if self.trainer.params.friend_cycles:
                x = torch.reshape(x, (len(batch), x.shape[1] * 3)) # [N, 3lstm_dim]

        # Classify features.
        x = self.head(x) # [N, num_labels]

        # Find out gold labels.
        gold_labels = torch.tensor(
            [
                self.trainer.label_map.get_label_id(node)
                for node in batch
            ],
            device=self.trainer.device
        ) # [num_labels]
        loss = self.loss(x, gold_labels)

        return ModelOutput(
            loss=loss,
            logits=x,
            gold_labels=gold_labels
        )
