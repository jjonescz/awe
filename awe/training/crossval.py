# 1. Set parameters in `data/params.json`.
# 2. Run `python -m awe.training.crossval`.

import argparse
import gc

import awe.data.set.pages
import awe.training.params
import awe.training.trainer


def main():
    args = parse_args()

    params = awe.training.params.Params.load_user(
        normalize=not args.print_max_index
    )
    if params is None:
        return
    if not args.print_max_index:
        print(f'{params=}')

    # Like FreeDOM and SimpDOM, use cyclic permutations.
    trainer = awe.training.trainer.Trainer(params)
    trainer.load_pretrained()
    trainer.load_dataset()

    total_count = len(trainer.ds.verticals[0].websites)
    if args.print_max_index:
        print(total_count - 1)
        return

    orig_name = params.version_name
    seed_len = len(params.train_website_indices)
    website_names = SWDE_VERTICAL_WEBSITES[params.vertical] \
        if params.dataset == awe.training.params.Dataset.swde \
        else [w.name for w in trainer.ds.verticals[0].websites]
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
        trainer.ds.clear_cache(awe.data.set.pages.ClearCacheRequest(
            labels=False
        ))
        gc.collect()

        trainer.init_features()
        trainer.split_data()
        trainer.create_dataloaders(create_test=True)
        trainer.create_model()
        trainer.create_version()
        trainer.train()
        trainer.test()

def get_cyclic_permutation_indices(seq_len: int, perm_idx: int, perm_len: int):
    return [(perm_idx + idx) % seq_len for idx in range(perm_len)]

def get_cyclic_permutation(seq: list[str], perm_idx: int, perm_len: int):
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
