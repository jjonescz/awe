import dataclasses
import enum
import json
import os
import warnings
from typing import Optional

import awe.data.constants
import awe.training.logging


class TokenizerFamily(str, enum.Enum):
    custom = 'custom'
    torchtext = 'torchtext' # tokenizer_id = 'basic_english'
    transformers = 'transformers' # tokenizer_id = 'bert-base-uncased'
    bert = 'bert'

class Dataset(str, enum.Enum):
    swde = 'swde'
    apify = 'apify'

@dataclasses.dataclass
class Params:
    """Hyper parameters with persistence."""

    # Dataset
    dataset: Dataset = Dataset.swde
    train_website_indices: list[int] = (0, 3, 4, 5, 7)
    """Only the first vertical for now."""
    train_subset: int = 2000
    val_subset: int = 50

    # Trainer
    epochs: int = 5
    version_name: str = ''
    restore_num: Optional[int] = None
    batch_size: int = 16
    save_every_n_epochs: Optional[int] = 1
    log_every_n_steps: int = 10
    eval_every_n_steps: Optional[int] = 50
    use_gpu: bool = True

    # Sampling
    load_visuals: bool = False
    classify_only_text_nodes: bool = False
    classify_only_variable_nodes: bool = False
    propagate_labels_to_leaves: bool = False
    validate_data: bool = True

    # Friend cycles
    friend_cycles: bool = False
    max_friends: int = 10

    # Word vectors
    tokenizer_family: TokenizerFamily = TokenizerFamily.custom
    tokenizer_id: str = ''
    tokenizer_fast: bool = True
    freeze_word_embeddings: bool = True
    pretrained_word_embeddings: bool = True

    # LSTM
    word_vector_function: Optional[str] = 'sum' # 'lstm', 'sum', 'mean'
    lstm_dim: int = 100
    lstm_args: Optional[dict] = None
    filter_node_words: bool = False

    # Word and char IDs
    cutoff_words: Optional[int] = 15
    """
    Maximum number of words to preserve in each node (or `None` to preserve
    all). Used by `CharacterIdentifiers` and `WordIdentifiers`.
    """

    cutoff_word_length: Optional[int] = 10
    """
    Maximum number of characters to preserve in each token (or `None` to
    preserve all). Used by `CharacterIdentifiers`.
    """

    # HTML DOM features
    tag_name_embedding: bool = False

    # Classifier
    learning_rate: float = 1e-3

    @classmethod
    def load_version(cls,
        version: awe.training.logging.Version,
        normalize: bool = False
    ):
        return cls.load_file(version.params_path, normalize=normalize)

    @classmethod
    def load_user(cls, normalize: bool = False):
        """Loads params from user-provided file."""
        path = f'{awe.data.constants.DATA_DIR}/params.json'
        if not os.path.exists(path):
            # Create file with default params as template.
            warnings.warn(f'No params file, creating one at {repr(path)}.')
            Params().save_file(path)
            return None
        return cls.load_file(path, normalize=normalize)

    @staticmethod
    def load_file(path: str, normalize: bool = False):
        with open(path, mode='r', encoding='utf-8') as f:
            result = Params(**json.load(f))

        if normalize:
            # Saving the params back adds default values of missing (new)
            # attributes and sorts attributes by key.
            result.save_file(path)

        return result

    def save_version(self, version: awe.training.logging.Version):
        self.save_file(version.params_path)

    def save_file(self, path: str):
        print(f'Saving {path!r}.')
        with open(path, mode='w', encoding='utf-8') as f:
            json.dump(dataclasses.asdict(self), f,
                indent=2,
                sort_keys=True
            )

    def as_dict(self, ignore_vars: list[str] = ()):
        d = dataclasses.asdict(self)
        for ignore_var in ignore_vars:
            d.pop(ignore_var, None)
        return d

    def as_set(self, ignore_vars: list[str] = ()):
        return set((k, repr(v)) for k, v in self.as_dict(ignore_vars).items())

    def difference(self, other: 'Params', ignore_vars: list[str] = ()):
        a = self.as_set(ignore_vars)
        b = other.as_set(ignore_vars)
        return a.symmetric_difference(b)
