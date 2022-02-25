from typing import TYPE_CHECKING

import torch
import transformers

import awe.features.feature
import awe.data.glove
import awe.data.graph.dom

if TYPE_CHECKING:
    import awe.features.extraction
    import awe.model.classifier


class WordIdentifiers(awe.features.feature.Feature):
    """Identifiers of word tokens. Used for pre-trained GloVe embeddings."""

    def __post_init__(self):
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.glove = awe.data.glove.LazyEmbeddings.get_or_create()

    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text)

    def get_token_id(self, token: str):
        # Indices start at 1; 0 is used for unknown and pad words.
        return self.glove.get_index(token, default=-1) + 1

    def prepare(self, node: awe.data.graph.dom.Node):
        params = self.trainer.params
        context = self.trainer.extractor.context

        # Find maximum word count.
        if node.is_text:
            counter = 0
            for i, _ in enumerate(self.tokenize(node.text)):
                if (
                    params.cutoff_words is not None and
                    i >= params.cutoff_words
                ):
                    break
                counter += 1
            context.max_num_words = max(context.max_num_words, counter)

    def compute(self, batch: 'awe.model.classifier.ModelInput'):
        context = self.trainer.extractor.context

        # Get word token indices.
        result = torch.zeros(len(batch), context.max_num_words,
            dtype=torch.int32,
            device=self.trainer.device,
        )
        for idx, node in enumerate(batch):
            if node.is_text:
                for i, token in enumerate(self.tokenize(node.text)):
                    if i >= context.max_num_words:
                        break
                    result[idx, i] = self.get_token_id(token)
        return result
