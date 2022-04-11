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
            dom_selector=lambda p: p.dom
        )

if __name__ == '__main__':
    main()
