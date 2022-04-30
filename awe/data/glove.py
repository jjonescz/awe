"""
Utilities for working with pre-trained GloVe embeddings.

To ensure GloVe embeddings are downloaded, run: `python -m awe.data.glove`.
"""

import os

from tqdm.auto import tqdm

import awe.data.constants

# HACK: Set `GENSIM_DATA_DIR` environment variable which is used by the `gensim`
# Python module to determine where to find/store the embeddings on disk.
GLOVE_DIR = f'{awe.data.constants.DATA_DIR}/glove'
os.environ['GENSIM_DATA_DIR'] = GLOVE_DIR

# pylint: disable=wrong-import-order, wrong-import-position
import gensim.downloader
import gensim.models

assert gensim.downloader.base_dir == GLOVE_DIR

# These constants determine which pre-trained embedding is loaded.
# See https://github.com/RaRe-Technologies/gensim-data.
VECTOR_DIMENSION = 100
MODEL_NAME = f'glove-wiki-gigaword-{VECTOR_DIMENSION}'

def disable_progress():
    """Disables progress bar of the `gensim.downloader` module."""

    # pylint: disable-next=protected-access
    gensim.downloader._progress = lambda *_, **__: None

def download_embeddings():
    """Downloads GloVe embeddings if they do not exist locally yet."""

    gensim.downloader.load(MODEL_NAME, return_path=True)

class LazyEmbeddings:
    """Container for lazily-loaded embeddings."""

    _model = None

    @classmethod
    def get_or_create(cls) -> gensim.models.KeyedVectors:
        if cls._model is None:
            with tqdm(total=1, desc='loading word vectors') as progress:
                cls._model = gensim.downloader.load(MODEL_NAME)
                progress.update()
        return cls._model

if __name__ == '__main__':
    download_embeddings()
