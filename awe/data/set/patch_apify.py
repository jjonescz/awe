# Run: `python -m awe.data.set.patch_apify`

from tqdm.auto import tqdm

import awe.data.set.apify

def main():
    # In Notino, `<noscript>` in head is not HTML compliant, so it gets parsed
    # wrong by Lexbor.
    ds = awe.data.set.apify.Dataset(only_websites=('notinoEn',), convert=False)
    w = ds.verticals[0].websites[0]
    for row in tqdm(w.df.iloc, desc='notinoEn'):
        html: str = row.localizedHtml
        html = html.replace('<noscript>&lt;link', '<noscript><link')
        html = html.replace('&gt;</noscript>', '></noscript>')
        row.localizedHtml = html
    w.save_json_df()

if __name__ == '__main__':
    main()
