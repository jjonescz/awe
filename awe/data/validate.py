# Run: `python -m awe.data.validate`

import argparse

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
args = parser.parse_args()

if args.dataset == 'apify':
    ds = awe.data.set.apify.Dataset(convert=False)
elif args.dataset == 'swde':
    ds = awe.data.set.swde.Dataset(suffix='-exact', convert=False)
pages = ds.get_all_pages(zip_websites=False)
validator = awe.data.validation.Validator(visuals=False)
validator.validate_pages(pages)
