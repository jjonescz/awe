import awe.data.set.pages

class Collater:
    """
    Prepares data for model. When called, takes batch of samples and returns
    batch of model inputs.
    """

    def __call__(self, samples: list[awe.data.set.pages.Page]):
        return []
