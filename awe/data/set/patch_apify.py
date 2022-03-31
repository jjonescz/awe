# Run: `python -m awe.data.set.patch_apify`

import pandas as pd
from tqdm.auto import tqdm

import awe.data.parsing
import awe.data.set.apify

ALZA_PARAMS_SELECTOR = '.params'


def main():
    patch_alza()
    patch_tesco()

def patch_alza():
    """
    Patches dataset `alzaEn`.

    1. Selector for category contains unnecessary space.
    2. Some `.params` don't exist on the page.
    2. Selector `.params` sometimes matches an empty node.
    """

    input_path = 'data/apify/alzaEn/augmented_dataset.json'
    print(f'Patching {input_path!r}...')
    df = pd.read_json(input_path)
    selector_category = df.columns.get_loc('selector_category')
    selector_specification = df.columns.get_loc('selector_specification')
    value_specification = df.columns.get_loc('specification')
    localized_html = df.columns.get_loc('localizedHtml')
    bug_1 = 0
    bug_2 = 0
    bug_3 = 0
    for idx in tqdm(range(len(df)), total=len(df), desc=input_path):
        # Bug 1.
        if df.iloc[idx, selector_category] == '.breadcrumbs .js-breadcrumbs':
            df.iloc[idx, selector_category] = '.breadcrumbs.js-breadcrumbs'
            bug_1 += 1

        if df.iloc[idx, selector_specification] == ALZA_PARAMS_SELECTOR:
            # Bug 2.
            removed_spec = False
            if df.iloc[idx, value_specification] == []:
                tree = awe.data.parsing.parse_html(df.iloc[idx, localized_html])
                if not tree.css_matches(ALZA_PARAMS_SELECTOR):
                    df.iloc[idx, selector_specification] = ''
                    df.iloc[idx, value_specification] = ''
                    bug_2 += 1
                    removed_spec = True

            # Bug 3.
            if not removed_spec:
                df.iloc[idx, selector_specification] = '#cpcm_cpc_mediaParams > .params'
                bug_3 += 1

    if bug_1 > 0 or bug_2 > 0:
        print(f'Saving {input_path!r} ({bug_1=} {bug_2=} {bug_3=})...')
        df.to_json(input_path)

def patch_tesco():
    """
    Patches dataset `tescoEn`, replacing `+` in a selector with `~` because the
    former crashes Lexbor (our HTML parser).
    """

    input_path = 'data/apify/tescoEn/augmented_dataset.json'
    print(f'Patching {input_path!r}...')
    df = pd.read_json(input_path)
    selector_specification = df.columns.get_loc('selector_specification')
    counter = 0
    for idx in tqdm(range(len(df)), total=len(df), desc=input_path):
        if df.iloc[idx, selector_specification] == '.product-info-block + section.tabularContent, div.product-info-block--undefined, div.product-info-block--undefined~.product-info-block':
            df.iloc[idx, selector_specification] = '.product-info-block ~ section.tabularContent, div.product-info-block--undefined, div.product-info-block--undefined~.product-info-block'
            counter += 1

    if counter > 0:
        print(f'Saving {input_path!r} ({counter=})...')
        df.to_json(input_path)

if __name__ == '__main__':
    main()
