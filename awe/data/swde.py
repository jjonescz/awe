from dataclasses import dataclass
import os
import re

SWDE_URL = 'https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip'
DATA_DIR = 'data'
SWDE_ZIP = f'{DATA_DIR}/swde.zip'
SWDE_DIR = f'{DATA_DIR}/swde'
SWDE_DATA_DIR = f'{SWDE_DIR}/data'

VERTICALS = [
    "auto",
    "book",
    "camera",
    "job",
    "movie",
    "nbaplayer",
    "restaurant",
    "university"
]

WEBSITE_REGEX = r"^(\w+)-(\w+)\((\d+)\)$"

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
    for subdir in os.listdir(f'{SWDE_DATA_DIR}/{vertical}'):
        website = Website(subdir)
        assert website.dir_name == subdir
        yield website
