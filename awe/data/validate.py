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
parser.add_argument('target',
    nargs='*',
    help='websites (Apify) or verticals (SWDE) to validate'
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
validator = awe.data.validation.Validator(visuals=False)
validator.validate_pages(pages)
