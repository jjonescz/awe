import os

from tqdm.auto import tqdm

import awe.data.constants

GLOVE_DIR = f'{awe.data.constants.DATA_DIR}/glove'
os.environ['GENSIM_DATA_DIR'] = GLOVE_DIR

# pylint: disable=wrong-import-order, wrong-import-position
import gensim.downloader
import gensim.models

assert gensim.downloader.base_dir == GLOVE_DIR

# See https://github.com/RaRe-Technologies/gensim-data.
VECTOR_DIMENSION = 100
MODEL_NAME = f'glove-wiki-gigaword-{VECTOR_DIMENSION}'

def download_embeddings():
    gensim.downloader.load(MODEL_NAME, return_path=True)

class LazyEmbeddings:
    """Container for lazily loaded model."""
    _model = None

    @classmethod
    def get_or_create(cls) -> gensim.models.KeyedVectors:
        if cls._model is None:
            with tqdm(total=1, desc='loading word vectors') as progress:
                cls._model = gensim.downloader.load(MODEL_NAME)
                progress.update()
        return cls._model
