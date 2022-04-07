#!/bin/bash

# Run: `./sh/crossval.sh`.

set -e

for i in $(seq 0 $(python -m awe.training.crossval --print-max-index 2>/dev/null))
do
    python -m awe.training.crossval -i $i -c 1
done
