import dataclasses
import enum
import json
import os
import warnings
from typing import Optional

import awe.data.constants
import awe.training.versioning


class TokenizerFamily(str, enum.Enum):
    custom = 'custom'
    torchtext = 'torchtext' # tokenizer_id = 'basic_english'
    transformers = 'transformers' # tokenizer_id = 'bert-base-uncased'
    bert = 'bert'

class Dataset(str, enum.Enum):
    swde = 'swde'
    apify = 'apify'

class VisualNeighborDistance(str, enum.Enum):
    center_point = 'center'
    rect = 'rect'

class AttentionNormalization(str, enum.Enum):
    vector = 'vector'
    softmax = 'softmax'

@dataclasses.dataclass
class Params:
    """Hyper parameters with persistence."""

    # Dataset
    dataset: Dataset = Dataset.swde
    vertical: str = 'auto'
    label_keys: list[str] = ('name', 'price', 'shortDescription', 'images')
    train_website_indices: list[int] = (0, 3, 4, 5, 7)
    """Only the first vertical for now."""
    exclude_websites: list[str] = ()
    train_subset: Optional[int] = 2000
    """Number of pages per website."""
    val_subset: Optional[int] = 50
    """Number of pages per website."""
    test_subset: Optional[int] = None
    """Number of pages per website."""

    # Trainer
    epochs: int = 5
    version_name: str = ''
    restore_num: Optional[int] = None
    batch_size: int = 16
    save_every_n_epochs: Optional[int] = 1
    save_better_val_loss_checkpoint: bool = True
    save_temporary_checkpoint: bool = True
    log_every_n_steps: int = 10
    eval_every_n_steps: Optional[int] = 50
    use_gpu: bool = True

    # Metrics
    exact_match: bool = False
    """
    Record also exact match metrics.

    Useful when `propagate_labels_to_leaves` is `True`.
    """

    # Sampling
    load_visuals: bool = False
    classify_only_text_nodes: bool = False
    classify_only_variable_nodes: bool = False
    propagate_labels_to_leaves: bool = False
    validate_data: bool = True
    none_cutoff: Optional[int] = None
    """
    From 0 to 100,000. The lower, the more non-target nodes will be sampled.
    """

    # Friend cycles
    friend_cycles: bool = False
    max_friends: int = 10

    # Visual neighbors
    visual_neighbors: bool = False
    n_neighbors: int = 4
    neighbor_distance: VisualNeighborDistance = VisualNeighborDistance.rect
    neighbor_normalize: Optional[AttentionNormalization] = AttentionNormalization.softmax

    # XPath
    xpath: bool = False

    # Ancestor chain
    ancestor_chain: bool = False
    n_ancestors: int = 5
    ancestor_function: Optional[str] = 'lstm' # 'lstm', 'attention', 'sum', 'mean'
    ancestor_lstm_out_dim: Optional[int] = None
    ancestor_lstm_args: Optional[dict[str]] = None
    ancestor_summarize: bool = False
    ancestor_dim: Optional[int] = None
    ancestor_normalize: Optional[AttentionNormalization] = AttentionNormalization.softmax

    # Word vectors
    tokenizer_family: TokenizerFamily = TokenizerFamily.custom
    tokenizer_id: str = ''
    tokenizer_fast: bool = True
    freeze_word_embeddings: bool = True
    pretrained_word_embeddings: bool = True
    tokenize_node_attrs: bool = False
    tokenize_node_attrs_only_ancestors: bool = True

    # LSTM
    word_vector_function: Optional[str] = 'sum' # 'lstm', 'sum', 'mean'
    lstm_dim: int = 100
    lstm_args: Optional[dict[str]] = None

    # Word and char IDs
    cutoff_words: Optional[int] = 15
    """
    Maximum number of words to preserve in each node (or `None` to preserve
    all). Used by `CharacterIdentifiers` and `WordIdentifiers`.
    """

    attr_cutoff_words: Optional[int] = 10

    cutoff_word_length: Optional[int] = 10
    """
    Maximum number of characters to preserve in each token (or `None` to
    preserve all). Used by `CharacterIdentifiers`.
    """

    # HTML DOM features
    tag_name_embedding: bool = False
    tag_name_embedding_dim: int = 30
    position: bool = False

    # Visual features
    enabled_visuals: Optional[list[str]] = None
    disabled_visuals: Optional[list[str]] = None

    # Classifier
    learning_rate: float = 1e-3
    label_smoothing: float = 0.0
    layer_norm: bool = False
    head_dims: list[int] = (128, 64)
    head_dropout: float = 0.5

    @classmethod
    def load_version(cls,
        version: awe.training.versioning.Version,
        normalize: bool = False
    ):
        return cls.load_file(version.params_path, normalize=normalize)

    @classmethod
    def load_user(cls, normalize: bool = False):
        """Loads params from user-provided file."""
        path = f'{awe.data.constants.DATA_DIR}/params.json'
        if not os.path.exists(path):
            # Create file with default params as template.
            warnings.warn(f'No params file, creating one at {path!r}.')
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

    def save_version(self, version: awe.training.versioning.Version):
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
