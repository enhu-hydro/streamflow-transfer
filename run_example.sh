#!/usr/bin/env bash
# End-to-end example: build the US arrays, run one configuration, collect.
# Takes about 10 minutes on one GPU once the data are in place.
set -e

SRC=${1:?usage: run_example.sh /path/to/1-camels/camels_daymetv2}

python prep_us.py --src "$SRC" --out data/us.npz

python train.py --npz data/us.npz --holdout dry_high --method erm --arch lstm \
                --seed 1 --epochs 60 --out runs/dry_high_erm_s1

python donor.py --npz data/us.npz --holdout dry_high --k 5 --out runs/donor

python score.py --runs runs --out figures
