import awe.data.set.label_validation
import awe.data.set.swde
import awe.data.set.apify

ds = awe.data.set.apify.Dataset(only_websites=('tescoEn',))
page = ds.verticals[0].websites[0].pages[0]
page_labels = page.get_labels()
page_dom = page.create_dom()
page_dom.init_nodes()
page_dom.root.parsed.css('.product-info-block + section.tabularContent')
