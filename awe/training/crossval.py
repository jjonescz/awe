"""
Script for cross-validation.

Usage:
1. Set parameters in `data/params.json`.
2. Run `python -m awe.training.crossval`.
3. Aggregate results via module `awe.training.crossval_mean`.
"""

import argparse

import awe.data.set.pages
import awe.training.params
import awe.training.trainer


def main():
    args = parse_args()

    params = awe.training.params.Params.load_user(
        normalize=not args.print_max_index
    )
    if params is None:
        print('Revisit params and re-run.')
        return
    if not args.print_max_index:
        print(f'{params=}')

    # Like FreeDOM and SimpDOM, use cyclic permutations.
    trainer = awe.training.trainer.Trainer(params)
    trainer.load_dataset()

    websites = trainer.ds.verticals[0].websites
    total_count = len(websites)
    if args.print_max_index:
        print(total_count - 1)
        return

    orig_name = params.version_name
    seed_len = len(params.train_website_indices)
    website_names = [w.name for w in websites]
    if params.dataset == awe.training.params.Dataset.swde:
        # Ensure ordering is consistent with SimpDOM.
        website_names = [name
            for name in SWDE_VERTICAL_WEBSITES[params.vertical]
            # Some websites might not be present in our dataset (because of bugs
            # in the SWDE dataset).
            if name in website_names
        ]
    print(f'{website_names=}, {seed_len=}')

    start_idx = args.index
    end_idx = args.index + (args.count or total_count)
    for perm_idx in range(start_idx, end_idx):
        trainer.params.version_name = f'{orig_name}-{perm_idx}'
        trainer.params.train_website_indices = get_cyclic_permutation_indices(
            seq_len=len(website_names),
            perm_idx=perm_idx,
            perm_len=seed_len
        )

        # Clear cache, because the whole dataset might not fit into memory and
        # we iteratively load everything across all cross-validation rounds.
        if args.clear_cache:
            num = trainer.ds.clear_cache(awe.data.set.pages.ClearCacheRequest(
                labels=False
            ))
            print(f'Cleared cache: {num:,}')

        trainer.init_features()
        trainer.split_data()
        trainer.create_dataloaders()
        trainer.create_model()
        trainer.create_version()
        trainer.train()
        trainer.extractor.enable_cache(False)
        trainer.test()

def get_cyclic_permutation_indices(seq_len: int, perm_idx: int, perm_len: int):
    """
    For a sequence of length `seq_len`, obtains `perm_idx`-th permutation of
    length `perm_len`.

    For example, if `seq_len=4`, `perm_len=3`,
    - `perm_idx=0` → `[0, 1, 2]`,
    - `perm_idx=1` → `[1, 2, 3]`,
    - `perm_idx=2` → `[2, 3, 0]`.
    """

    return [(perm_idx + idx) % seq_len for idx in range(perm_len)]

def get_cyclic_permutation(seq: list[str], perm_idx: int, perm_len: int):
    """
    For a sequence `seq`, obtains `perm_idx`-th permutation of length
    `perm_len`.

    Simply takes elements from `seq` according to
    `get_cyclic_permutation_indices`.
    """

    indices = get_cyclic_permutation_indices(
        seq_len=len(seq),
        perm_idx=perm_idx,
        perm_len=perm_len
    )
    return [seq[idx] for idx in indices]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Runs cross-validation training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--print-max-index',
        dest='print_max_index',
        action='store_true',
        default=False,
        help='only print max index and exit'
    )
    parser.add_argument('--clear-cache',
        dest='clear_cache',
        action='store_true',
        default=False,
        help='clear DOM cache before each fold'
    )
    parser.add_argument('-i',
        dest='index',
        type=int,
        default=0,
        help='start index'
    )
    parser.add_argument('-c',
        dest='count',
        type=int,
        help='max count'
    )
    return parser.parse_args()

# Ordering from SimpDOM source code.
SWDE_VERTICAL_WEBSITES = {
    "auto": [
        "msn", "aol", "kbb", "cars", "yahoo", "autoweb", "autobytel",
        "automotive", "carquotes", "motortrend"
    ],
    "book": [
        "abebooks", "amazon", "barnesandnoble", "bookdepository",
        "booksamillion", "borders", "buy", "christianbook", "deepdiscount",
        "waterstones"
    ],
    "camera": [
        "amazon", "beachaudio", "buy", "compsource", "ecost", "jr", "newegg",
        "onsale", "pcnation", "thenerds"
    ],
    "job": [
        "careerbuilder", "dice", "hotjobs", "job", "jobcircle", "jobtarget",
        "monster", "nettemps", "rightitjobs", "techcentric"
    ],
    "movie": [
        "allmovie", "amctv", "boxofficemojo", "hollywood", "iheartmovies",
        "imdb", "metacritic", "msn", "rottentomatoes", "yahoo"
    ],
    "nbaplayer": [
        "espn", "fanhouse", "foxsports", "msnca", "nba", "si", "slam",
        "usatoday", "wiki", "yahoo"
    ],
    "restaurant": [
        "fodors", "frommers", "gayot", "opentable", "pickarestaurant",
        "restaurantica", "tripadvisor", "urbanspoon", "usdiners", "zagat"
    ],
    "university": [
        "collegeboard", "collegenavigator", "collegeprowler", "collegetoolkit",
        "ecampustours", "embark", "matchcollege", "princetonreview",
        "studentaid", "usnews"
    ]
}

if __name__ == '__main__':
    main()
