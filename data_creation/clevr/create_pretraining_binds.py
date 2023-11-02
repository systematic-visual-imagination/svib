import os
import json

import argparse

import itertools as it

from macros import *


def pretraining_bind_generator(counts):

    all_binds = set(it.product(*[list(range(count)) for count in counts]))

    return all_binds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', default='precomputed_binds')

    args = parser.parse_args()

    # Make Binds
    binds = pretraining_bind_generator([len(COLORS), len(SHAPES), len(SIZES), len(MATERIALS)])

    # Make directory
    data_path = args.save_path
    os.makedirs(data_path, exist_ok=True)

    with open(os.path.join(data_path, "pretraining_binds.json"), 'w') as outfile:
        json.dump({
            "binds": list(binds)
        }, outfile)
