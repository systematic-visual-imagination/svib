import os
import json
import math
import random

import argparse

import itertools as it

from macros import *


def bind_generator(counts, train_alphas, test_alpha):

    assert max(train_alphas) <= 1. - test_alpha, "Improper Train/Test split ratio"

    num_props = len(counts)
    max_count = max(counts)

    core_binds = set([tuple([i % counts[j] for j in range(num_props)]) for i in range(max_count)])

    all_binds = set(it.product(*[list(range(count)) for count in counts]))

    noncore_binds = all_binds.difference(core_binds)
    num_noncore_binds = len(noncore_binds)

    num_test_binds = math.floor(num_noncore_binds * test_alpha)
    test_binds = set(random.sample(noncore_binds, num_test_binds))

    noncore_nontest_binds = noncore_binds.difference(test_binds)

    train_binds = {}
    for train_alpha in train_alphas:
        num_noncore_train_binds = math.ceil(num_noncore_binds * train_alpha)
        train_binds[train_alpha] = set(random.sample(noncore_nontest_binds, num_noncore_train_binds))
        train_binds[train_alpha] = train_binds[train_alpha].union(core_binds)

    return test_binds, train_binds, core_binds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', default='precomputed_binds')

    parser.add_argument('--test_alpha', type=float, default=0.2)
    parser.add_argument('--train_alphas', nargs='+', default=[0.01 * i for i in range(81)])

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    # Set Seeds
    random.seed(args.seed)

    # Make Binds
    test_binds, train_binds, core_binds = bind_generator([len(SHAPES), len(SIZES), len(MATERIALS)], train_alphas=args.train_alphas, test_alpha=args.test_alpha)

    # Make directory
    data_path = args.save_path
    os.makedirs(data_path, exist_ok=True)

    with open(os.path.join(data_path, "test_binds_alpha_{:.2f}.json".format(args.test_alpha)), 'w') as outfile:
        json.dump({
            "binds": list(test_binds)
        }, outfile)

    with open(os.path.join(data_path, "core_binds.json"), 'w') as outfile:
        json.dump({
            "binds": list(core_binds)
        }, outfile)

    for alpha, binds in train_binds.items():
        with open(os.path.join(data_path, "train_binds_alpha_{:.2f}.json".format(alpha)), 'w') as outfile:
            json.dump({
                "binds": list(binds)
            }, outfile)
