"""
Script for finding variable nodes in all pages in a dataset.

Run: `python -m awe.data.find_variable_nodes <target>`.
"""

import argparse

from tqdm.auto import tqdm

import awe.data.set.apify
import awe.data.set.pages
import awe.data.set.swde


def main():
    args = parse_args()

    if args.target == 'apify':
        ds = awe.data.set.apify.Dataset()
    else:
        ds = awe.data.set.swde.Dataset(
            suffix='-exact',
            only_verticals=(args.target,),
        )

    for website in (p := tqdm(ds.verticals[0].websites)):
        website: awe.data.set.pages.Website
        p.set_description(website.name)
        website.find_variable_xpaths()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute variable nodes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('target',
        help='either "apify" or SWDE website name'
    )
    return parser.parse_args()

if __name__ == '__main__':
    main()
