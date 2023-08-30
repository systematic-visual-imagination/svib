"""Contains code for data set creation as well as live environments."""

import numpy as np
import h5py
from tqdm import tqdm
import random
import argparse

from spriteworld import renderers as spriteworld_renderers
from spriteworld.sprite import Sprite
import os
import cv2

from distinctipy import distinctipy

import math
import json

from utils import generate_binds_with
from create import checker, apply_rule, RULES, generate_gt, sort_and_render
from distutils.dir_util import copy_tree

# Constant
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)


if __name__ == '__main__':
    ''' Unit testing'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--save_path', default='data/')
    parser.add_argument('--test_path', default='test')
    parser.add_argument('--num_train', type=int, default=64000)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--rule', default='CP')

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    test_path = args.test_path

    with open(os.path.join(test_path, "info.json"), 'r') as outfile:
        train_info = json.load(outfile)

    COLORS = train_info["colors"]
    SHAPES = train_info["shapes"]
    SIZES = train_info["sizes"]
    test_ratio = train_info["ratio"]
    UNSEEN_BINDS = train_info["unseen_binds"]
    CORES = train_info["cores"]

    # PARAMETERS
    C, H, W = 3, args.image_size, args.image_size

    run_num = args.num_train
    N_min, N_max = 2, 2
    Att_N = len(SHAPES)

    # Composition space ratio
    train_ratio = args.train_ratio
    assert train_ratio <= 1-test_ratio, "train/test split ratio"

    # PARAMETERS per TASK
    R = args.rule
    params = {
            "motion": (0.3, 0),
            "colors": COLORS,
            "sizes": SIZES,
            "shapes": SHAPES,
            }

    BINDS = generate_binds_with(COLORS, SHAPES, SIZES, UNSEEN_BINDS, CORES,  ratio = train_ratio) 


    save_path = os.path.join(args.save_path, f"dsprites-{R}-alpha-{train_ratio}")
    copy_tree(test_path, os.path.join(save_path, "test"))

    train_path = os.path.join(save_path, f"train")
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

    print(f"Train Comp: {len(BINDS)}, Test Comp: {len(UNSEEN_BINDS)}")

    # Renderer
    renderer = spriteworld_renderers.PILRenderer(
            image_size=(H, W),
        anti_aliasing=10,
        bg_color='black'
    )

    print("Train Set : {}".format(run_num))
    for run in tqdm(range(run_num)):
        sample_path = os.path.join(train_path, f"{run:08d}")
        os.makedirs(sample_path, exist_ok=True)

        N = np.random.choice(np.arange(N_min, N_max + 1))
        bind = np.random.choice(np.arange(len(BINDS)), size=(N))
        bind = BINDS[bind]

        color = bind[:,0]
        shape = bind[:,1]
        size = bind[:,2]
        
        size = np.expand_dims(np.array(SIZES)[size], axis=1)
        x = np.random.random(size=(N, 2))

        while True:
            x = np.random.random(size=(N, 2)) * (1 - 2 * 0.4 * size) + 0.4 * size
            if checker(x, size * 0.4):
                break


        angle = np.random.random(size=(N)) * 0
        
        sprites = []
        for i in range(N):
            s = Sprite(x[i][0], x[i][1],
                       shape=SHAPES[shape[i]],
                       c0=int(COLORS[color[i]][0]),
                       c1=int(COLORS[color[i]][1]),
                       c2=int(COLORS[color[i]][2]),
                       angle=angle[i],
                       scale=size[i][0])
            sprites += [s]

        img = sort_and_render(renderer, sprites, COLORS)
        cv2.imwrite(os.path.join(sample_path, f"source.png"), img)

        # save GT information
        objects = generate_gt(sprites)

        scene_struct_source = {
        'image_filename': os.path.basename(sample_path),
        'objects': objects,
        }

        with open(os.path.join(sample_path, 'source.json'), 'w') as f:
            json.dump(scene_struct_source, f, indent=2)
        
        sprites = apply_rule(sprites, R, params)

        target_img = sort_and_render(renderer, sprites, COLORS)
        cv2.imwrite(os.path.join(sample_path, f"target.png"), target_img)

        # save GT information
        objects = generate_gt(sprites)
        
        scene_struct_target = {
        'image_filename': os.path.basename(sample_path),
        'objects': objects,
        }

        with open(os.path.join(sample_path, 'target.json'), 'w') as f:
            json.dump(scene_struct_target, f, indent=2)

    print('Dataset saved at : {}'.format(save_path))
