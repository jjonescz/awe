"""`doit` configuration file; see https://pydoit.org/contents.html."""

import os
import subprocess

from awe.data import glove, swde

DOIT_CONFIG = {
    'verbosity': 2
}

def shell(command: str):
    print(f'$ {command}')
    subprocess.run(command, shell=True, check=True)

def task_download_swde():
    """download SWDE dataset"""

    return {
        'actions': [
            f'mkdir -p {os.path.dirname(swde.ZIP)}',
            f'curl {swde.URL} --output {swde.ZIP}'
        ],
        'targets': [swde.ZIP],
        'uptodate': [True]
    }

def task_extract_swde():
    """extract SWDE zip"""

    return {
        'actions': [f'unzip {swde.ZIP} -d {swde.DIR} || true'],
        'file_dep': [swde.ZIP],
        'targets': [swde.DIR]
    }

def task_extract_swde_src():
    """extract SWDE sourceCode.zip"""

    input_zip = f'{swde.DIR}/sourceCode/sourceCode.zip'
    output_dir = f'{swde.DIR}/src'
    return {
        'actions': [f'unzip {input_zip} -d {output_dir}'],
        'file_dep': [input_zip],
        'targets': [output_dir]
    }

def task_extract_swde_7z():
    """extract SWDE 7z archives"""

    input_dir = f'{swde.DIR}/src'
    names = [v for v in swde.VERTICAL_NAMES] + [swde.GROUND_TRUTH]
    input_zips = [f'{input_dir}/{n}.7z' for n in names]
    output_dir = swde.DATA_DIR
    output_dirs = [f'{output_dir}/{n}' for n in names]
    def extract_src():
        for archive in input_zips:
            shell(f'7z x {archive} -aos -o"{output_dir}" >/dev/null')

    return {
        'actions': [
            f'mkdir -p {output_dir}',
            extract_src
        ],
        'file_dep': input_zips,
        'targets': output_dirs
    }

def task_download_glove():
    """download pre-trained GloVe embeddings"""

    return {
        'actions': [glove.download_embeddings],
        'targets': [glove.GLOVE_DIR],
        'uptodate': [True]
    }
