from awe import awe_graph

class Collater:
    """
    Prepares data for model. When called, takes batch of samples and returns
    batch of model inputs.
    """

    def __call__(self, samples: list[awe_graph.HtmlPage]):
        return []
