import dataclasses
from typing import TYPE_CHECKING

import torch
import transformers
import transformers.activations
from transformers.models.big_bird.modeling_big_bird import \
    BigBirdForQuestionAnsweringModelOutput
from transformers.modeling_outputs import \
    BaseModelOutputWithPoolingAndCrossAttentions

if TYPE_CHECKING:
    import awe.qa.trainer

ModelOutput = BigBirdForQuestionAnsweringModelOutput

@dataclasses.dataclass
class Prediction:
    batch: transformers.BatchEncoding
    outputs: ModelOutput

class Model(torch.nn.Module):
    def __init__(self,
        params: 'awe.qa.trainer.TrainerParams',
        model: transformers.BigBirdForQuestionAnswering,
    ):
        super().__init__()
        self.params = params
        self.model = model

        output_size = model.config.hidden_size # 768
        input_size = output_size

        if params.add_boundary_info:
            input_size += 2

        self.dropout1 = torch.nn.Dropout(0.1)
        self.dense1 = torch.nn.Linear(input_size, 3072)
        self.intermediate_act_fn = transformers.activations.gelu_fast
        self.dense2 = torch.nn.Linear(3072, 768)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.layer_norm = torch.nn.LayerNorm(768, eps=1e-12)
        self.qa_outputs = torch.nn.Linear(768, 2)
        # Last token is used for truncated answers, hence it's ignored.
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def configure_optimizers(self):
        return transformers.AdamW(self.parameters(), lr=1e-5)

    def forward(self, batch: transformers.BatchEncoding) -> ModelOutput:
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.model.bert(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )

        x = outputs.last_hidden_state # [batch, seq_length, output_size]

        if self.params.add_boundary_info:
            node_starts = batch.node_starts # [batch, seq_length]
            node_starts = node_starts.unsqueeze(2) # [batch, seq_length, 1]
            node_ends = batch.node_ends.unsqueeze(2)
            x = torch.concat((x, node_starts, node_ends), dim=-1)
                # [batch, seq_length, input_size]

        x = self.dropout1(x)
        x = self.dense1(x) # [batch, seq_length, 3072]
        x = self.intermediate_act_fn(x)
        x = self.dense2(x) # [batch, seq_length, 768]
        x = self.dropout2(x)
        x = self.layer_norm(x)
        logits = self.qa_outputs(x) # [batch, seq_length, 2]

        if self.params.mask_questions:
            # Effectively ignore question tokens.
            logits[batch.question_mask, :] = 0

        start_logits = logits[:, :, 0] # [batch, seq_length]
        end_logits = logits[:, :, 1] # [batch, seq_length]

        start_loss = self.loss_fn(start_logits, batch.start_positions)
        end_loss = self.loss_fn(end_logits, batch.end_positions)
        total_loss = (start_loss + end_loss) / 2

        return ModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits
        )
