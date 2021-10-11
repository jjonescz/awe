import os
import re
from dataclasses import dataclass, field

import constants

URL = 'https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip'
ZIP = f'{constants.DATA_DIR}/swde.zip'
DIR = f'{constants.DATA_DIR}/swde'
DATA_DIR = f'{DIR}/data'

WEBSITE_REGEX = r'^(\w+)-(\w+)\((\d+)\)$'

def ignore_field(**kwargs):
    return field(init=False, repr=False, hash=False, compare=False, **kwargs)

@dataclass
class Vertical:
    name: str
    _websites: list['Website'] = ignore_field(default=None)

    @property
    def websites(self):
        if self._websites is None:
            self._websites = [w for w in get_websites(self.name)]
        return self._websites

@dataclass
class Website:
    vertical: str
    name: str
    page_count: int

    def __init__(self, dir_name: str):
        match = re.search(WEBSITE_REGEX, dir_name)
        self.vertical = match.group(1)
        self.name = match.group(2)
        self.page_count = int(match.group(3))

    @property
    def dir_name(self):
        return f'{self.vertical}-{self.name}({self.page_count})'

def get_websites(vertical: str):
    for subdir in os.listdir(f'{DATA_DIR}/{vertical}'):
        website = Website(subdir)
        assert website.dir_name == subdir
        yield website

VERTICALS = [
    Vertical('auto'),
    Vertical('book'),
    Vertical('camera'),
    Vertical('job'),
    Vertical('movie'),
    Vertical('nbaplayer'),
    Vertical('restaurant'),
    Vertical('university')
]
