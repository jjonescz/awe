from awe.data import swde
from awe import utils

sds = swde.Dataset(suffix='-exact')
pages = [p for v in sds.verticals for w in v.websites for p in w.pages]
print(utils.summarize_pages(pages))
