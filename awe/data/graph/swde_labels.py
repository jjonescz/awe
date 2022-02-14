import dataclasses
import re
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import awe.data.graph.swde

GROUNDTRUTH_REGEX = r'^(\w+)-(\w+)-(\w+)\.txt$'

@dataclasses.dataclass
class GroundtruthFile:
    """
    Represents the file with groundtruth key-value pairs for an SWDE website.
    """

    website: 'awe.data.graph.swde.Website'
    label_key: str
    entries: list['GroundtruthEntry']

    def __init__(self, website: 'awe.data.graph.swde.Website', file_name: str):
        self.website = website
        match = re.search(GROUNDTRUTH_REGEX, file_name)
        assert match.group(1) == website.vertical.name
        assert match.group(2) == website.name
        self.label_key = match.group(3)
        self.entries = list(self._iterate_entries())

    @property
    def file_name(self):
        return f'{self.website.vertical.name}-{self.website.name}-{self.label_key}.txt'

    @property
    def file_path(self):
        return f'{self.website.vertical.groundtruth_dir}/{self.file_name}'

    def _iterate_entries(self):
        with open(self.file_path, mode='r', encoding='utf-8-sig') as file:
            lines = [line.rstrip('\r\n') for line in file]

        # Read first line.
        vertical, site, label_key = lines[0].split('\t')
        assert vertical == self.website.vertical.name
        assert site == self.website.name
        assert label_key == self.label_key

        # Read second line.
        count, _, _, _ = lines[1].split('\t')
        assert int(count) == self.website.page_count

        # Read rest of the file.
        for index, line in enumerate(lines[2:]):
            expected_index, expected_nonnull_count, *values = line.split('\t')
            assert int(expected_index) == index
            parsed_values = [] if values == ['<NULL>'] else values
            assert int(expected_nonnull_count) == len(parsed_values)
            page = self.website.get_page_at(index)
            if page is None:
                warnings.warn(
                    f'No page at {index} in {repr(self.website.dir_name)}.')
            else:
                assert page.index == index
            yield GroundtruthEntry(self, page, parsed_values)

@dataclasses.dataclass
class GroundtruthEntry:
    file: GroundtruthFile
    page: 'awe.data.graph.swde.Page'

    label_values: list[str]
    """Label values as loaded from the groundtruth `field` file."""
