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

from create import generate_unseen_binds, checker, apply_rule, RULES, generate_gt
from distutils.dir_util import copy_tree

def generate_binds_with(colors_, shapes_, sizes_, binds_, cores_, ratio = 0.8):
    must = max([len(colors_), len(shapes_), len(sizes_)])

    colors = np.append(np.arange(len(colors_)), np.random.randint(low = len(colors_), size = must - len(colors_)))
    shapes = np.append(np.arange(len(shapes_)), np.random.randint(low = len(shapes_), size = must - len(shapes_)))
    sizes = np.append(np.arange(len(sizes_)), np.random.randint(low = len(sizes_), size = must - len(sizes_)))

    N = len(colors_)*len(shapes_)*len(sizes_)
    N_sample = math.floor((N - must) * ratio)

    binds_ = np.concatenate([binds_, cores_], axis=0)

    if N_sample > N - len(binds_): # switch to i.i.d
        print("switch to i.i.d")
        return binds_[:]

    unseen_binds_ = []

    unseen_binds_ = np.array(np.meshgrid(colors, shapes, sizes)).T.reshape(-1,3)
   
    rows1 = unseen_binds_.view([('', unseen_binds_.dtype)] * unseen_binds_.shape[1])
    rows2 = binds_.view([('', binds_.dtype)] * binds_.shape[1])
    unseen_binds_ = np.setdiff1d(rows1, rows2).view(unseen_binds_.dtype).reshape(-1, unseen_binds_.shape[1])
    
    binds = cores_[:]

    for i in range(N_sample):
        idx = np.random.choice(np.arange(unseen_binds_.shape[0]))
        sample = unseen_binds_[idx]
        unseen_binds_ = np.delete(unseen_binds_, idx, axis=0)
        binds += [sample]

    return np.array(binds)



# Constant
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)


if __name__ == '__main__':
    ''' Unit testing'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--save_path', default='data/')
    parser.add_argument('--num_train', type=int, default=64000)
    parser.add_argument('--num_test', type=int, default=12800)
    parser.add_argument('--rule_name', default="rule")

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)  

    train_ratio = 0.0
    R = args.rule_name

    while train_ratio <= 0.7:
        train_ratio = round(train_ratio, 1)
        dir_name = f"tmp/data/dsprites-{R}-alpha-{train_ratio}"
        train_path = os.path.join(dir_name, "train")

        with open(os.path.join(train_path, "info.json"), 'r') as outfile:
            train_info = json.load(outfile)

        COLORS = train_info["colors"]
        SHAPES = train_info["shapes"]
        SIZES = train_info["sizes"]

        train_ratio = train_info["ratio"]
        BINDS = np.array(train_info["binds"])
        CORES = train_info["cores"]

        # PARAMETERS
        C, H, W = 3, args.image_size, args.image_size

        run_num = args.num_train
        N_min, N_max = 2, 2
        Att_N = len(SHAPES)


        # PARAMETERS per TASK
        params = {
                "motion": (0.3, 0),
                "colors": COLORS,
                "sizes": SIZES,
                "shapes": SHAPES,
                }

        save_path = os.path.join(args.save_path, f"dsprites-{R}-alpha-{train_ratio}")

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

        print(f"Train Comp: {len(BINDS)}")

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


            angle = np.random.random(size=(N)) * 360
            
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

            img = renderer.render(sprites)
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

            target_img = renderer.render(sprites)
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

        
        if train_ratio != 0.0:
            copy_tree(f"data/dsprites-{R}-alpha-0.0/test", os.path.join(save_path, "test"))

        else:
            test_path = os.path.join(dir_name, "test")

            with open(os.path.join(test_path, "info.json"), 'r') as outfile:
                test_info = json.load(outfile)

            COLORS = test_info["colors"]
            SHAPES = test_info["shapes"]
            SIZES = test_info["sizes"]

            test_ratio = test_info["ratio"]
            UNSEEN_BINDS = np.array(test_info["unseen_binds"])
            CORES = test_info["cores"]

            run_num_test = args.num_test
            N_min, N_max = 2, 2
            Att_N = len(SHAPES)

            # PARAMETERS per TASK
            params = {
                    "motion": (0.3, 0),
                    "colors": COLORS,
                    "sizes": SIZES,
                    "shapes": SHAPES,
                    }

           
            test_path = os.path.join(save_path, f"test")
            os.makedirs(test_path, exist_ok=True)

            test_info = {
                    "colors": COLORS,
                    "shapes": SHAPES,
                    "sizes": SIZES,
                    "ratio": test_ratio,
                    "unseen_binds": UNSEEN_BINDS.tolist(),
                    "cores": CORES,
                    }

            print(test_info)

            with open(os.path.join(test_path, "info.json"), 'w') as outfile:
                json.dump(test_info, outfile)

            print(f"Test Comp: {len(UNSEEN_BINDS)}")

            print("Test Set : {}".format(run_num_test))

            for run in tqdm(range(run_num_test)):
                sample_path = os.path.join(test_path, f"{run:08d}")
                os.makedirs(sample_path, exist_ok=True)

                N = np.random.choice(np.arange(N_min, N_max + 1))
                bind = np.random.choice(np.arange(len(UNSEEN_BINDS)), size=(N))
                bind = UNSEEN_BINDS[bind]

                color = bind[:,0]
                shape = bind[:,1]
                size = bind[:,2]
                
                size = np.expand_dims(np.array(SIZES)[size], axis=1)
                x = np.random.random(size=(N, 2))

                while True:
                    x = np.random.random(size=(N, 2)) * (1 - 2 * 0.4 * size) + 0.4 * size
                    if checker(x, size * 0.4):
                        break


                angle = np.random.random(size=(N)) * 360

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

                img = renderer.render(sprites)
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

                target_img = renderer.render(sprites)
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

        train_ratio += 0.2
