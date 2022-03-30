# Run: `python -m awe.data.validate`

import argparse
import os
import warnings

import awe.data.set.apify
import awe.data.set.swde
import awe.data.validation

# Parse CLI arguments.
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
    help='file with list of page paths that will be considered'
)
parser.add_argument('--save-back',
    dest='save_back',
    action='store_true',
    help='saves list of invalid pages back to `--read-list`',
    default=False
)
parser.add_argument('--save-list',
    dest='save_list',
    help='saves list of invalid pages to this text file'
)
parser.add_argument('-q',
    dest='quiet',
    action='store_true',
    help='suppress warnings',
    default=False
)
parser.add_argument('--max-pages',
    dest='max_pages',
    type=int,
    help='maximum number of pages to validate'
)
parser.add_argument('--skip-pages',
    dest='skip_pages',
    type=int,
    help='number of pages to skip at the beginning'
)
parser.add_argument('--max-errors',
    dest='max_errors',
    type=int,
    help='maximum number of errors before quitting the validation'
)
parser.add_argument('--visuals',
    dest='visuals',
    action='store_true',
    help='validate visuals',
    default=False
)
parser.add_argument('--no-labels',
    dest='no_labels',
    action='store_true',
    help='do not validate labels',
    default=False
)
parser.add_argument('--convert',
    dest='convert',
    action='store_true',
    help='work with SQLite database rather than JSON',
    default=False
)
args = parser.parse_args()

# Validate arguments.
if args.save_back and args.read_list is None:
    raise ValueError(
        'Argument `--save-back` cannot be specified without `--read-list`.')

# Open dataset.
if len(args.target) == 0:
    args.target = None
if args.dataset == 'apify':
    ds = awe.data.set.apify.Dataset(
        only_websites=args.target,
        convert=args.convert
    )
elif args.dataset == 'swde':
    ds = awe.data.set.swde.Dataset(
        suffix='-exact',
        only_verticals=args.target,
        convert=args.convert
    )

# Get its pages.
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

# Slice pages.
if args.skip_pages is not None:
    pages = pages[args.skip_pages:]
if args.max_pages is not None:
    pages = pages[:args.max_pages]

validator = awe.data.validation.Validator(
    labels=not args.no_labels,
    visuals=args.visuals
)

# Write invalid pages to a file (iteratively during the validation).
if args.save_list is not None:
    validator.write_invalid_to(args.save_list)

# Validate.
def validate():
    validator.validate_pages(pages, max_invalid=args.max_errors)
    print(f'Validation complete (invalid={validator.num_invalid}, ' +
        f'pages={len(pages)}).')
if args.quiet:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module=r'awe\.data\.validation')
        warnings.filterwarnings('ignore', module=r'awe\.data\.visual\.dom')
        validate()
else:
    validate()

# Update list of invalid pages.
if args.save_back:
    num_original = len(page_paths)
    num_valid = 0
    num_invalid = 0
    for page in pages:
        if page.valid is True:
            num_valid += 1
            page_paths.discard(page.original_html_path)
        elif page.valid is False:
            num_invalid += 1
            page_paths.add(page.original_html_path)
    sorted_paths = sorted(page_paths)
    with open(args.read_list, mode='w', encoding='utf-8', newline='\n') as f:
        f.writelines(f'{p}\n' for p in sorted_paths)
    print(
        f'Saved {len(sorted_paths)} to {args.read_list!r} ' +
        f'({num_original=}, {num_valid=}, {num_invalid=}).'
    )

# Save new list of invalid pages.
if args.save_list is not None:
    validator.file.close()
    print(f'Saved {validator.num_invalid} to {args.save_list!r}.')
