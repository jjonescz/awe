# `doit` configuration file; see https://pydoit.org/contents.html.

import urllib.request
import pathlib
import os
import shutil
import sys

DOIT_CONFIG = {
    'verbosity': 2
}
SWDE_URL = 'https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip'
DATA_DIR = 'data'
SWDE_ZIP = f'{DATA_DIR}/swde.zip'
SWDE_DIR = f'{DATA_DIR}/swde'

def task_install():
    """pip install"""

    return {
        'actions': ['./sh/install.sh'],
        'file_dep': ['requirements.txt']
    }

def task_download_swde():
    """download SWDE dataset"""

    def report(blocknr, blocksize, size):
        # Inspired by https://stackoverflow.com/a/21363808.
        current = blocknr * blocksize
        sys.stdout.write("\r{0:.2f}%".format(100.0 * current / size))

    def download():
        pathlib.Path(os.path.dirname(SWDE_ZIP)).mkdir(parents=True, exist_ok=True)
        print(f'Downloading {SWDE_URL} to {SWDE_ZIP}')
        urllib.request.urlretrieve(SWDE_URL, SWDE_ZIP, report)
        print()

    return {
        'actions': [download],
        'targets': [SWDE_ZIP],
        'uptodate': [True]
    }

def task_extract_swde():
    """extract SWDE zip"""

    def extract():
        shutil.unpack_archive(SWDE_ZIP, SWDE_DIR)

    return {
        'actions': [extract],
        'file_dep': [SWDE_ZIP],
        'targets': [SWDE_DIR]
    }
