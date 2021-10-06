# `doit` configuration file; see https://pydoit.org/contents.html.

import os
import glob
import subprocess
import awe.data.swde

DOIT_CONFIG = {
    'verbosity': 2
}
SWDE_URL = 'https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip'
DATA_DIR = 'data'
SWDE_ZIP = f'{DATA_DIR}/swde.zip'
SWDE_DIR = f'{DATA_DIR}/swde'

def exec(command: str):
    print(f'$ {command}')
    subprocess.run(command, shell=True, check=True)

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

def task_extract_swde_src():
    """extract SWDE sourceCode.zip"""

    input = f'{SWDE_DIR}/sourceCode/sourceCode.zip'
    output = f'{SWDE_DIR}/src'
    return {
        'actions': [f'unzip {input} -d {output}'],
        'file_dep': [input],
        'targets': [output]
    }

def task_extract_swde_verticals():
    """extract SWDE 7z archives"""

    inputs = [f'{SWDE_DIR}/src/{v}.7z' for v in awe.data.swde.VERTICALS]
    output_dir = f'{SWDE_DIR}/data'
    outputs = [f'{output_dir}/{v}' for v in awe.data.swde.VERTICALS]
    def extract_src():
        for archive in glob.glob(f'{input}/*.7z'):
            exec(f'7z x {archive} -o"{output_dir}"')

    return {
        'actions': [
            f'mkdir -p {output_dir}',
            extract_src
        ],
        'file_dep': inputs,
        'targets': outputs
    }
