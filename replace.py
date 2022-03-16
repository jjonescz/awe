from tqdm.auto import tqdm

import awe.data.set.pages
import awe.data.set.swde

ds = awe.data.set.swde.Dataset(
    suffix='-exact',
    only_verticals=('job',),
    convert=False
)
pages = ds.get_all_pages()
with open('data/swde/invalid_pages.txt', mode='r', encoding='utf-8') as f:
    page_paths = { l.rstrip() for l in f.readlines() }
pages = [
    p for p in pages
    if p.original_html_path in page_paths
]
print(f'Found {len(pages)} of {len(page_paths)} pages in the list.')
for page in tqdm(pages, desc='pages'):
    page: awe.data.set.pages.Page
    with open(page.html_path, 'r', encoding='utf-8-sig') as f:
        html_text = f.read()
    html_text = html_text.replace('â€“', '–')
    with open(page.html_path, 'w', encoding='utf-8-sig') as f:
        f.write(html_text)
