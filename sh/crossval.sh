#!/bin/bash

# Run: `./sh/crossval.sh`.
# First (optional) argument: start index of cross-validation fold (0-9).
# Second (optional) argument: end index of cross-validation fold (0-9).

set -e

START_INDEX=${1:-0}
END_INDEX=${2:-$(python -m awe.training.crossval --print-max-index 2>/dev/null)}
echo "Running cross-validation from $START_INDEX to $END_INDEX..."
for i in $(seq $START_INDEX $END_INDEX)
do
    date
    python -m awe.training.crossval -i $i -c 1
done
date
