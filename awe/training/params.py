"""
Training parameters and their (de)serialization.

To create new user parameters (saved at `data/params.json`) with default values
or validate existing, run as module: `python -m awe.training.params`.
"""

import dataclasses
import enum
import json
import os
import warnings
from typing import Optional

import awe.data.constants
import awe.training.versioning


class TokenizerFamily(str, enum.Enum):
    """
    See `awe.features.text.WordIdentifiers.__post_init__` for details about the
    tokenizers used.
    """

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

def _freeze(value):
    """
    Creates field with a mutable default value.

    Python doesn't like this being done directly. But `Params` are never
    mutated anyway, so this workaround is easiest.
    """

    return dataclasses.field(default_factory=lambda: value)

@dataclasses.dataclass
class Params:
    """
    All training hyperparameters including control of feature extraction and
    data loading.
    """

    # Dataset
    dataset: Dataset = Dataset.swde
    """Dataset to load."""

    vertical: str = 'auto'
    """
    Vertical to load.

    Only considered when using the SWDE dataset for now.
    """

    label_keys: list[str] = ()
    """
    Set of keys to select from the dataset.

    Only considered when using the Apify dataset for now.
    """

    train_website_indices: list[int] = (0, 1, 2, 3, 4)
    """
    Indices of websites to put in the training set.

    Note that this is automatically changed during cross-validation experiments.
    """

    exclude_websites: list[str] = ()
    """Website names to exclude from loading."""

    train_subset: Optional[int] = 100
    """Number of pages per website to use for training."""

    val_subset: Optional[int] = 5
    """
    Number of pages per website to use for validation (evaluation after each
    training epoch).
    """

    test_subset: Optional[int] = 250
    """
    Number of pages per website to use for testing (evaluation of each
    cross-validation run).
    """

    # Trainer
    epochs: int = 3
    """
    Number of epochs (passes over all training samples) to train the model for.
    """

    version_name: str = ''
    """
    Name used to save version in the logdir (see `awe.training.versioning`).
    """

    restore_num: Optional[int] = None
    """Existing version number to restore."""

    batch_size: int = 32
    """Number of samples to have in a mini-batch during training/evaluation."""

    save_every_n_epochs: Optional[int] = None
    """How often a checkpoint should be saved."""

    save_better_val_loss_checkpoint: bool = False
    """
    Save checkpoint after each epoch when better validation loss is achieved.
    """

    save_temporary_checkpoint: bool = True
    """
    Save checkpoint after each epoch and then delete it when a new epoch is
    trained.

    This is useful when training "interactively", i.e., one is able to cancel
    the training and still have the last checkpoint saved, but at the same time
    the storage space is not wasted by saving all checkpoints.
    """

    log_every_n_steps: int = 100
    """
    How often to evaluate and write metrics to TensorBoard during training.
    """

    eval_every_n_steps: Optional[int] = None
    """
    How often to execute full evaluation pass on the validation subset during
    training.

    This is useful when the training set has lots of samples and one wants to
    see how validation metrics change also during one epoch.
    """

    use_gpu: bool = True
    """Train on GPU (if available, otherwise a warning is issued), or CPU."""

    # Metrics
    exact_match: bool = False
    """
    Record also exact match metrics.

    Useful when `propagate_labels_to_leaves` is `True`.
    """

    # Sampling
    load_visuals: bool = True
    """
    When loading HTML for pages, also load JSON visuals, parse visual attributes
    and attach them to DOM nodes in memory.
    """

    classify_only_text_nodes: bool = True
    """Sample only text fragments."""

    classify_only_variable_nodes: bool = False
    """
    Sample only variable text fragments.

    Variable nodes are determined as in SimpDOM (see
    `Website.find_variable_xpaths`).
    """

    classify_also_html_tags: list[str] = ()
    """
    Apart from text fragments (if `classify_only_text_nodes` is `True`), these
    HTML tags are also sampled.
    """

    propagate_labels_to_leaves: bool = False
    """Propagate labels from inner nodes to all their leaf descendants."""

    validate_data: bool = True
    """Validate sampled page DOMs."""

    ignore_invalid_pages: bool = True
    """Of sampled and validated pages, ignore those that are invalid."""

    none_cutoff: Optional[int] = 30_000
    """
    From 0 to 100,000. The higher, the more non-target nodes will be sampled.
    """

    # Friend cycles
    friend_cycles: bool = False
    """Use friend cycle feature as defined in SimpDOM."""

    max_friends: int = 10
    """Number of friends in the friend cycle."""

    # Visual neighbors
    visual_neighbors: bool = True
    """Use visual neighbors as a feature when classifying nodes."""

    n_neighbors: int = 10
    """Number of visual neighbors (the closes ones)."""

    neighbor_distance: VisualNeighborDistance = VisualNeighborDistance.rect
    """How to determine the closes visual neighbors."""

    neighbor_normalize: Optional[AttentionNormalization] = AttentionNormalization.vector
    """
    How to normalize neighbor distances before feeding them to the attention
    module.
    """

    normalize_distance: bool = False
    """
    Whether to normalize neighbor distances (absolute pixels) dividing them by
    page width (obtaining relative distances approximately in range [0, 1]).
    """

    # Ancestor chain
    ancestor_chain: bool = True
    """Use DOM ancestors as a feature when classifying nodes."""

    n_ancestors: Optional[int] = None
    """`None` to use all ancestors."""

    ancestor_lstm_out_dim: int = 10
    """Output dimension of the LSTM aggregating ancestor features."""

    ancestor_lstm_args: Optional[dict[str]] = _freeze({'bidirectional': True})
    """
    Additional keyword arguments to the LSTM layer aggregating ancestor
    features.
    """

    ancestor_tag_dim: Optional[int] = 30
    """Embedding dimension of ancestor HTML tag names."""

    xpath: bool = False
    """
    Like ancestor chain but only HTML tags and without limit on number of
    ancestors. This is separate so it can be used alongside limited ancestor
    chain.
    """

    # Word vectors
    tokenizer_family: TokenizerFamily = TokenizerFamily.bert
    """Which tokenizer to use."""

    tokenizer_id: str = ''
    """Specific tokenizer ID (available only for some tokenizer families)."""

    tokenizer_fast: bool = True
    """
    Use fast HuggingFace tokenizers (implemented in Rust).

    Only considered if `tokenizer_family == "transformers"`.
    """

    freeze_word_embeddings: bool = True
    """Avoid training word embedding matrix."""

    pretrained_word_embeddings: bool = True
    """
    Load pre-trained GloVe embeddings (otherwise, they are randomly
    initialized).
    """

    # HTML attributes
    tokenize_node_attrs: list[str] = ('itemprop',)
    """
    DOM attributes to tokenize and use as a feature when classifying nodes and
    also as a feature of each ancestor in the ancestor chain.

    Common choices are `itemprop`, `id`, `name`, `class`.
    """

    tokenize_node_attrs_only_ancestors: bool = True
    """Use the DOM attribute feature only for nodes of the ancestor chain."""

    # LSTM
    word_vector_function: Optional[str] = 'lstm'
    """
    How to aggregate word vectors.

    Can be `lstm`, or one of PyTorch functions, e.g., `sum` or `mean`.
    """

    lstm_dim: int = 100
    """Output dimension of the LSTM aggregating word vectors."""

    lstm_args: Optional[dict[str]] = _freeze({'bidirectional': True})
    """
    Additional keyword arguments to the LSTM layer aggregating word vectors.
    """

    # Word and char IDs
    cutoff_words: Optional[int] = 15
    """
    Maximum number of words to preserve in each node (`None` to preserve all).

    Used when tokenizing node text for word LSTM.
    """

    attr_cutoff_words: Optional[int] = 10
    """Like `cutoff_words` but for the DOM attribute feature."""

    cutoff_word_length: Optional[int] = 10
    """
    Maximum number of characters to preserve in each token (`None` to preserve
    all).

    Used when embedding characters for CNN.
    """

    # HTML DOM features
    tag_name_embedding: bool = True
    """
    Whether to use HTML tag name as a feature when classifying nodes.

    See `awe.features.dom.HtmlTag`.
    """

    tag_name_embedding_dim: int = 30
    """Dimension of the output vector of HTML tag name embedding."""

    position: bool = True
    """
    Whether to use visual position as a feature when classifying nodes.

    See `awe.features.dom.Position`.
    """

    # Visual features
    enabled_visuals: Optional[list[str]] = (
        "font_size", "font_style", "font_weight", "font_color"
    )
    """
    Filter visual attributes to only those in this list.

    For the list of all attributes, see `awe.data.visual.attribute`.
    """

    disabled_visuals: Optional[list[str]] = None
    """
    Filter visual attributes to not contain those in this list.

    For the list of all attributes, see `awe.data.visual.attribute`.
    """

    # Classifier
    learning_rate: float = 1e-3
    """Learning rate of the Adam optimizer."""

    weight_decay: float = 0.0 # e.g., 0.0001
    """Weight decay of the Adam optimizer."""

    label_smoothing: float = 0.0 # e.g., 0.1
    """Label smoothing of the cross-entropy loss."""

    layer_norm: bool = False
    """Use layer normalization in the classification head."""

    head_dims: list[int] = (100, 10)
    """Dimensions of feed-forward layers in the classification head."""

    head_dropout: float = 0.3
    """Dropout probability in the classification head."""

    gradient_clipping: Optional[float] = None
    """Clip gradients before backpropagation."""

    @classmethod
    def load_version(cls,
        version: awe.training.versioning.Version,
        normalize: bool = False
    ):
        """Loads parameters of the specified `version`."""

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
        """
        Loads parameters from JSON file at `path`.

        If `normalize` is `True`, the file is saved back (with normalized
        indentation, ordering or properties, and also the set of parameters if
        it changes).
        """

        with open(path, mode='r', encoding='utf-8') as f:
            result = Params(**json.load(f))

        if normalize:
            # Saving the params back adds default values of missing (new)
            # attributes and sorts attributes by key.
            result.save_file(path)

        return result

    def save_version(self, version: awe.training.versioning.Version):
        """Saves parameters to directory of the specified `version`."""

        self.save_file(version.params_path)

    def save_file(self, file_path: str):
        """Saves parameters as JSON file at `file_path`."""

        print(f'Saving {file_path!r}.')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode='w', encoding='utf-8') as f:
            json.dump(dataclasses.asdict(self), f,
                indent=2,
                sort_keys=True
            )

    def as_dict(self, ignore_vars: list[str] = ()):
        """Converts this object to dictionary."""

        d = dataclasses.asdict(self)
        for ignore_var in ignore_vars:
            d.pop(ignore_var, None)
        return d

    def as_set(self, ignore_vars: list[str] = ()):
        """Converts this object to set of key-value pairs."""

        return set((k, repr(v)) for k, v in self.as_dict(ignore_vars).items())

    def difference(self, other: 'Params', ignore_vars: list[str] = ()):
        """Finds difference to `other` parameters."""

        a = self.as_set(ignore_vars)
        b = other.as_set(ignore_vars)
        return a.symmetric_difference(b)

    def patch_for_inference(self):
        """Ensures some paramaters are set correctly for inference."""

        self.validate_data = False
        self.classify_only_variable_nodes = False

if __name__ == '__main__':
    print(Params.load_user(normalize=True))
