# Computes mean metrics after running `crossval` several times.
# Run: `python -m awe.training.crossval_mean <first_version_num>`.

import argparse
import json
import os

import numpy as np

import awe.training.params
import awe.training.versioning


def main():
    args = parse_args()

    params = awe.training.params.Params.load_user()

    # Load all saved metrics.
    all_metrics = []
    idx = 0
    while True:
        version = awe.training.versioning.Version(
            number=args.version_num,
            name=f'{params.version_name}-{idx}'
        )
        if not version.exists():
            break
        results_path = version.get_results_path('test')
        if not os.path.exists(results_path):
            break
        with open(results_path, mode='r', encoding='utf-8') as f:
            all_metrics.append(json.load(f))

    # Compute mean metrics.
    keys = { k for m in all_metrics for k in m.keys() }
    all_values = {
        k: [v for m in all_metrics if (v := m.get(k)) is not None]
        for k in keys
    }
    mean_metrics = { k: np.mean(vs) for k, vs in all_values.items() }
    metric_counts = { k: len(vs) for k, vs in all_values.items() }
    print('Mean metrics:')
    print(json.dumps(mean_metrics, indent=2, sort_keys=True))
    print('Counts:')
    print(json.dumps(metric_counts, indent=2, sort_keys=True))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluates cross-validation training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('version_num',
        type=int,
        help='first cross-validation version number'
    )
    return parser.parse_args()

if __name__ == '__main__':
    main()
