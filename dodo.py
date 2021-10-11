# `doit` configuration file; see https://pydoit.org/contents.html.

import os
import subprocess
import awe.data.swde

DOIT_CONFIG = {
    'verbosity': 2
}
SWDE_URL = awe.data.swde.SWDE_URL
DATA_DIR = awe.data.swde.DATA_DIR
SWDE_ZIP = awe.data.swde.SWDE_ZIP
SWDE_DIR = awe.data.swde.SWDE_DIR
SWDE_DATA_DIR = awe.data.swde.SWDE_DATA_DIR
SWDE_VERTICALS = awe.data.swde.VERTICALS

def shell(command: str):
    print(f'$ {command}')
    subprocess.run(command, shell=True, check=True)

def task_configure():
    """configure"""

    script = './sh/configure.sh'
    return {
        'actions': [script],
        'file_dep': [script]
    }

def task_install():
    """pip install"""

    script = './sh/install.sh'
    return {
        'actions': [script],
        'file_dep': [script, 'requirements.txt']
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

    input_zip = f'{SWDE_DIR}/sourceCode/sourceCode.zip'
    output_dir = f'{SWDE_DIR}/src'
    return {
        'actions': [f'unzip {input_zip} -d {output_dir}'],
        'file_dep': [input_zip],
        'targets': [output_dir]
    }

def task_extract_swde_verticals():
    """extract SWDE 7z archives"""

    input_dir = f'{SWDE_DIR}/src'
    input_zips = [f'{input_dir}/{v.name}.7z' for v in SWDE_VERTICALS]
    output_dir = SWDE_DATA_DIR
    output_dirs = [f'{output_dir}/{v.name}' for v in SWDE_VERTICALS]
    def extract_src():
        for archive in input_zips:
            shell(f'7z x {archive} -o"{output_dir}"')

    return {
        'actions': [
            f'mkdir -p {output_dir}',
            extract_src
        ],
        'file_dep': input_zips,
        'targets': output_dirs
    }
