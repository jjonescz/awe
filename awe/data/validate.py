# Run: `python -m awe.data.validate`

import argparse
import os
import warnings

import awe.data.set.apify
import awe.data.set.swde
import awe.data.validation

parser = argparse.ArgumentParser(
    description='Validates datasets',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('-d',
    dest='dataset',
    choices=['apify', 'swde'],
    default='apify',
    help='dataset'
)
parser.add_argument('target',
    nargs='*',
    help='websites (Apify) or verticals (SWDE) to validate'
)
parser.add_argument('--read-list',
    dest='read_list',
    help='File with list of page paths that will be considered'
)
args = parser.parse_args()

if args.dataset == 'apify':
    ds = awe.data.set.apify.Dataset(only_websites=args.target, convert=False)
elif args.dataset == 'swde':
    ds = awe.data.set.swde.Dataset(
        suffix='-exact',
        only_verticals=args.target,
        convert=False
    )

pages = ds.get_all_pages(zip_websites=False)

# Keep only pages from previous list of invalid pages.
if args.read_list is not None:
    if os.path.exists(args.read_list):
        with open(args.read_list, mode='r', encoding='utf-8') as f:
            page_paths = { l.rstrip() for l in f.readlines() }
        pages = [
            p for p in pages
            if p.original_html_path in page_paths
        ]
        print(
            f'Found {len(pages)} of {len(page_paths)} pages in the list ' +
            f'{args.read_list!r}.'
        )
    else:
        warnings.warn(f'List file not found ({args.read_list!r}).')

validator = awe.data.validation.Validator(visuals=False)
validator.validate_pages(pages)
