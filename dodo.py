# `doit` configuration file; see https://pydoit.org/contents.html.

import os

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

    return {
        'actions': [
            f'mkdir -p {os.path.dirname(SWDE_ZIP)}',
            f'curl {SWDE_URL} --output {SWDE_ZIP}'
        ],
        'targets': [SWDE_ZIP],
        'uptodate': [True]
    }

def task_extract_swde():
    """extract SWDE zip"""

    return {
        'actions': [f'unzip {SWDE_ZIP} -d {SWDE_DIR} || true'],
        'file_dep': [SWDE_ZIP],
        'targets': [SWDE_DIR]
    }
