import dataclasses

import awe.qa.parser
from awe import awe_graph


@dataclasses.dataclass
class Sample:
    page: awe_graph.HtmlPage
    label: str
    values: list[str]

def get_samples(pages: list[awe_graph.HtmlPage]):
    """
    Expands each page into multiple training samples (one question-answer per
    label).
    """

    label_maps = [awe.qa.parser.get_page_labels(page) for page in pages]
    return [
        Sample(page, label, values)
        for page, label_map in zip(pages, label_maps)
        for label, values in label_map.items()
        if len(values) != 0
    ]
