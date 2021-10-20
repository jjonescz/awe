import os

from awe.data import constants

GLOVE_DIR = f'{constants.DATA_DIR}/glove'
os.environ['GENSIM_DATA_DIR'] = GLOVE_DIR

from gensim import downloader as api
from gensim import models

assert api.base_dir == GLOVE_DIR

# See https://github.com/RaRe-Technologies/gensim-data.
MODEL_NAME = 'glove-wiki-gigaword-100'

def download_embeddings():
    api.load(MODEL_NAME, return_path=True)

def get_embeddings() -> models.KeyedVectors:
    return api.load(MODEL_NAME)
