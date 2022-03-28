# Run: `python -m awe.data.set.patch_apify`

from tqdm.auto import tqdm

import awe.data.set.apify

def main():
    # In Notino, `<noscript>` in head is not HTML compliant, so it gets parsed
    # wrong by Lexbor.
    ds = awe.data.set.apify.Dataset(only_websites=('notinoEn',))
    w = ds.verticals[0].websites[0]
    for page in tqdm(w.pages, desc='notinoEn'):
        page: awe.data.set.apify.Page
        html = page.get_html_text()
        html = html.replace('<noscript>&lt;link', '<noscript><link')
        html = html.replace('&gt;</noscript>', '></noscript>')
        w.db.replace(page.index, page.url, html, w.db.get_visuals(page.index), w.db.get_metadata(page.index))
    w.db.save()

if __name__ == '__main__':
    main()
