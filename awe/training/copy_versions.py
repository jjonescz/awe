"""
Copies versions from source directory to target directory and re-numbers to
start at the specified number.

Run: `python -m awe.training.copy_versions <source_dir> <target_dir> <start_num>`
"""

import argparse
import os
import shutil

import awe.training.versioning


def main():
    args = parse_args()

    # Gather source versions.
    sources = [
        source
        for source_subdir in os.listdir(args.source_dir)
        if (source := awe.training.versioning.Version.try_parse(source_subdir))
    ]
    sources.sort(key=lambda v: v.number)

    # Copy and re-number.
    num = args.start_num

    for source in sources:
        source_subdir = os.path.join(args.source_dir, source.version_dir_name)
        target = awe.training.versioning.Version(num, source.name)
        target_subdir = os.path.join(args.target_dir, target.version_dir_name)
        if args.dry:
            print(f'Would copy {source_subdir!r} -> {target_subdir!r}.')
        else:
            shutil.copytree(source_subdir, target_subdir)

        num += 1

    if not args.dry:
        print(f'Copied {len(sources)} directories.')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Copies and renumbers versions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('source_dir',
        help='source logdir (containing version subdirectories)'
    )
    parser.add_argument('target_dir',
        help='target logdir'
    )
    parser.add_argument('start_num',
        type=int,
        nargs='?',
        default=1,
        help='where to start numbering versions copied to the target directory'
    )
    parser.add_argument('-n',
        dest='dry',
        action='store_true',
        help='dry mode, only print what would be done'
    )
    return parser.parse_args()

if __name__ == '__main__':
    main()
