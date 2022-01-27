from typing import Optional

import transformers


class Pipeline:
    _model: Optional[transformers.BigBirdForQuestionAnswering] = None
    _tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None

    def __init__(self,
        model_id = 'vasudevgupta/bigbird-roberta-natural-questions'
    ):
        self.model_id = model_id

    def load(self):
        _ = self.model
        _ = self.tokenizer

    @property
    def model(self) -> transformers.BigBirdForQuestionAnswering:
        if self._model is None:
            self._model = transformers.AutoModelForQuestionAnswering \
                .from_pretrained(self.model_id)
        return self._model

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        if self._tokenizer is None:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id)
        return self._tokenizer
