"""Contains code for data set creation as well as live environments."""

import numpy as np
import h5py
from tqdm import tqdm
import random
import argparse

from spriteworld.sprite import Sprite

import os
import cv2

import math
import json

from macros import *
from utils import *
from rule import RULES, apply_rule
from create_data import checker, generate_samples

from distutils.dir_util import copy_tree

if __name__ == '__main__':
    ''' Unit testing'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--save_path', default='data/')
    parser.add_argument('--test_path', default='test')
    parser.add_argument('--num_train', type=int, default=64000)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--rule', default='CP')
    parser.add_argument('--pretrain', default=False, action='store_true')

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    test_path = args.test_path

    with open(os.path.join(test_path, "info.json"), 'r') as outfile:
        train_info = json.load(outfile)

    COLORS = train_info["colors"]
    SHAPES = train_info["shapes"]
    SIZES = train_info["sizes"]

    UNSEEN_BINDS = train_info["unseen_binds"]
    CORES = train_info["cores"]

    # PARAMETERS
    C, H, W = 3, args.image_size, args.image_size

    N_min, N_max = 2, 2

    # Composition space ratio
    train_ratio = args.train_ratio
    test_ratio = train_info["ratio"]
    assert train_ratio <= 1-test_ratio, "train/test split ratio"

    # PARAMETERS per TASK
    R = args.rule
    if R not in RULES:
        print("Undefined Rule")
        exit()

    # Renderer
    renderer = spriteworld_renderers.PILRenderer(
            image_size=(H, W),
            anti_aliasing=10,
            bg_color='black'
            )

    BINDS = generate_binds_with(COLORS, SHAPES, SIZES, CORES, UNSEEN_BINDS, ratio = train_ratio) 

    print(f"Comp Num: {len(BINDS)}")

    save_path = os.path.join(args.save_path, f"dsprites-{R}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #copy_tree(test_path, os.path.join(save_path, "test"))

    train_path = os.path.join(save_path, f"train/alpha-{train_ratio}")
    os.makedirs(train_path, exist_ok=True)

    train_info = {
            "colors": COLORS,
            "shapes": SHAPES,
            "sizes": SIZES,
            "ratio": train_ratio,
            "binds": BINDS.tolist(),
            "cores": CORES,
            }

    print(train_info)

    with open(os.path.join(train_path, "info.json"), 'w') as outfile:
        json.dump(train_info, outfile)

    
    generate_samples(train_path, args.num_train, N_min, N_max, R, BINDS, CORES, renderer, args)
