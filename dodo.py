# `doit` configuration file; see https://pydoit.org/contents.html.

import urllib.request
import pathlib
import os

DOIT_CONFIG = {
    'verbosity': 2
}
SWDE_URL = 'https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip'
SWDE_ZIP = 'data/swde.zip'

def task_install():
    """pip install"""

    return {
        'actions': ['./sh/install.sh'],
        'file_dep': ['requirements.txt']
    }

def task_download():
    """download SWDE dataset"""

    def download():
        pathlib.Path(os.path.dirname(SWDE_ZIP)).mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SWDE_URL, SWDE_ZIP)

    return {
        'actions': [download],
        'targets': [SWDE_ZIP],
        'uptodate': [True]
    }
