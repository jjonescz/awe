DOIT_CONFIG = {
    'verbosity': 2
}

def task_install():
    """pip install"""

    return {
        'actions': ['./sh/install.sh'],
        'file_dep': ['requirements.txt']
    }
