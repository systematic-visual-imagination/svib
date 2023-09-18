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


def checker(positions, sizes):
    N = positions.shape[0]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dist = np.sqrt(np.sum((positions[i] - positions[j]) * (positions[i] - positions[j])))
            min_dist = sizes[i] + sizes[j]
            if dist < min_dist:
                return False
    return True

def generate_samples(path, num_samples, N_min, N_max, rule, binds, cores, renderer, args):

    print("Train Set : {}".format(num_samples))

    for run in tqdm(range(num_samples)):
        sample_path = os.path.join(path, f"{run:08d}")
        os.makedirs(sample_path, exist_ok=True)

        N = np.random.choice(np.arange(N_min, N_max + 1))
        bind = np.random.choice(np.arange(len(binds)), size=(N))
        bind = binds[bind]

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
        
        pairs = []
        for i in range(N):
            s = Sprite(x[i][0], x[i][1],
                       shape=SHAPES[shape[i]],
                       c0=int(COLORS[color[i]][0]),
                       c1=int(COLORS[color[i]][1]),
                       c2=int(COLORS[color[i]][2]),
                       angle=angle[i],
                       scale=size[i][0])
            mask_c = MASK_COLORS[i]

            pairs += [[s, mask_c]]

        img, mask = sort_and_render(renderer, pairs, COLORS)
        cv2.imwrite(os.path.join(sample_path, f"source.png"), img)
        cv2.imwrite(os.path.join(sample_path, f"source_mask.png"), mask)


        sprites = [pair[0] for pair in pairs]
        # save GT information
        objects = generate_gt(sprites)

        scene_struct_source = {
        'image_filename': os.path.basename(sample_path),
        'objects': objects,
        }

        with open(os.path.join(sample_path, 'source.json'), 'w') as f:
            json.dump(scene_struct_source, f, indent=2)

        if not args.pretrain:
            # apply rule
            apply_rule(sprites, rule)
            
            target_img, target_mask = sort_and_render(renderer, pairs, COLORS)
            cv2.imwrite(os.path.join(sample_path, f"target.png"), target_img)
            cv2.imwrite(os.path.join(sample_path, f"target_mask.png"), target_mask)

            # save GT information
            objects = generate_gt(sprites)
            
            scene_struct_target = {
            'image_filename': os.path.basename(sample_path),
            'objects': objects,
            }

            with open(os.path.join(sample_path, 'target.json'), 'w') as f:
                json.dump(scene_struct_target, f, indent=2)

    print('Dataset saved at : {}'.format(path))


if __name__ == '__main__':
    ''' Unit testing'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--save_path', default='data/')

    parser.add_argument('--num_train', type=int, default=64000)
    parser.add_argument('--num_test', type=int, default=12800)

    parser.add_argument('--train_ratios', type=float, nargs='+', default=[0.2 * i for i in range(4)])
    parser.add_argument('--test_ratio', type=float, default=0.2)

    parser.add_argument('--rule', default='CP')

    parser.add_argument('--pretrain', default=False, action='store_true')

    args = parser.parse_args()
    
    random.seed(0)
    np.random.seed(0)

    # PARAMETERS
    C, H, W = 3, args.image_size, args.image_size
 
    N_min, N_max = 2, 2

    # Composition space ratio
    test_ratio = args.test_ratio
    train_ratios = args.train_ratios

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

    save_path = os.path.join(args.save_path, f"dsprites-{R}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    BINDS = []
    
    for train_ratio in train_ratios:

        assert train_ratio <= 1-test_ratio, "train/test split ratio"
        
        # Binding Pairs (TRAIN)
        if len(BINDS) == 0:
            BINDS, CORES = generate_binds(COLORS, SHAPES, SIZES, BINDS, ratio = train_ratio)
        else:
            BINDS, _ = generate_binds(COLORS, SHAPES, SIZES, BINDS, ratio = train_ratio)
        
        print(f"Comp Num: {len(BINDS)}")

        train_path = os.path.join(save_path, f"train/alpha-{train_ratio}")
        os.makedirs(train_path, exist_ok=True)

        train_info = {
                "colors": COLORS,
                "shapes": SHAPES,
                "sizes": SIZES,
                "ratio": train_ratio,
                "binds": BINDS.tolist(),
                "cores": CORES.tolist(),
                }

        print(train_info)
        with open(os.path.join(train_path, "info.json"), 'w') as outfile:
            json.dump(train_info, outfile)
            
        generate_samples(train_path, args.num_train, N_min, N_max, R, BINDS, CORES, renderer, args)

    if not args.pretrain:

        UNSEEN_BINDS = generate_unseen_binds(COLORS, SHAPES, SIZES, BINDS, ratio = test_ratio)  # test
        
        print(f"Comp Num: {len(UNSEEN_BINDS)}")

        test_path = os.path.join(save_path, f"test")
        os.makedirs(test_path, exist_ok=True)
        
        test_info = {
                "colors": COLORS,
                "shapes": SHAPES,
                "sizes": SIZES,
                "ratio": test_ratio,
                "unseen_binds": UNSEEN_BINDS.tolist(),
                "cores": CORES.tolist(),
                }

        print(test_info)
        with open(os.path.join(test_path, "info.json"), 'w') as outfile:
            json.dump(test_info, outfile)

        generate_samples(test_path, args.num_test, N_min, N_max, R, UNSEEN_BINDS, CORES, renderer, args)
