# Run: `python -m awe.data.validate`

import awe.data.validation
import awe.data.set.apify

ds = awe.data.set.apify.Dataset(convert=False)
pages = ds.get_all_pages(zip_websites=False)
awe.data.validation.validate(pages, visuals=False)
