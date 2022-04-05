import awe.data.set.apify

ds = awe.data.set.apify.Dataset(
    only_websites=('conradEn',),
    only_label_keys=('name', 'price', 'shortDescription', 'images')
)
page = next(p for p in ds.get_all_pages() if p.html_path ==
    'data/apify/conradEn/pages/localized_html_https-www-conrad-com-p-makita-bo4565j-random-orbit-sander-200-w-2317163.htm')
page.cache_dom()
page.dom.init_nodes()
page_visuals = page.load_visuals()
page_visuals.fill_tree_light(page.dom)
page.dom.filter_nodes()
page.dom.init_labels(propagate_to_leaves=False)
print('Done')
