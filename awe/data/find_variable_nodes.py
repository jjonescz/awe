# Run: `python -m awe.data.find_variable_nodes`.

from tqdm.auto import tqdm

import awe.data.set.pages
import awe.data.set.swde


def main():
    ds = awe.data.set.swde.Dataset(
        suffix='-exact',
        only_verticals=('auto',),
        convert=True,
    )
    for website in (p := tqdm(ds.verticals[0].websites)):
        website: awe.data.set.pages.Website
        p.set_description(website.name)
        website.get_variable_xpaths(
            dom_selector=get_page_dom
        )

def get_page_dom(page: awe.data.set.pages.Page):
    page_dom = page.dom
    page_dom.init_nodes()
    page_dom.init_labels()
    return page_dom

if __name__ == '__main__':
    main()
