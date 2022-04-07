#!/bin/bash

# Run: `./sh/crossval.sh`.

set -e

for i in 0 1 2 3 4 5 6 7 8 9 10
do
    python -m awe.training.crossval -i $i -c 1
done
