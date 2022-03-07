import json
import os
from typing import Any


class DomData:
    """Can load visual attributes saved by `extractor.ts`."""

    data: dict[str, Any] = None

    def __init__(self, path: str):
        self.path = path

    def exists(self):
        return os.path.exists(self.path)

    def get_json_str(self):
        with open(self.path, mode='r', encoding='utf-8') as file:
            return file.read()

    def load_json(self):
        """Reads DOM data from JSON."""
        self.data = json.loads(self.get_json_str())
