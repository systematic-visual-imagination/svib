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

from utils import generate_binds, generate_unseen_binds

# Constant
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)

RULES = ["None", "CP", "CC", "ShAC", "ShAASh", "CAASi", "MIntra", "MInter", "MMix1", "MMix2", "NCP", "NShAC", "NShAASh", "NCAASi", "MNInter", "MNIntra", "MNMix1", "MNMix2", "MOOD", "MFull", "NA", "colorwithshape", "colorwithposition", "poschangecolor"]

def apply_rule(sprites, rule, params):
    N = len(sprites)

    if rule == "CP":
        for i in range(N):
            sprites[i].move(params["motion"])

    elif rule == "CC":
        COLORS = params["colors"]
        colors = np.array([(COLORS.index(list(sprite._color)) + 1) % len(COLORS) for sprite in sprites])
        for i in range(N):
            sprites[i]._color = tuple(COLORS[colors[i]])

    elif rule == "ShAASh":
        SHAPES = params["shapes"]

        shapes = []

        for sprite in sprites:
            shapes.append(SHAPES.index(sprite._shape))

        shapes = np.roll(shapes, 1)

        for i in range(N):
            sprites[i]._shape = SHAPES[shapes[i]]

            sprites[i]._reset_centered_path()

    elif rule == "CAASi":
        COLORS = params["colors"]
        SIZES = params["sizes"]

        colors = []

        for sprite in sprites:
            colors.append(COLORS.index(list(sprite._color)))

        sizes = np.roll(colors, 1)

        for i in range(N):
            sprites[i]._scale = SIZES[sizes[i]]

            sprites[i]._reset_centered_path()
    
    elif rule == "ShAC":
        COLORS = params["colors"]
        SHAPES = params["shapes"]

        for sprite in sprites:
            sprite._color = tuple(COLORS[SHAPES.index(sprite._shape) % len(COLORS)])

    elif rule == "MIntra":
        # CP & ShAC
        COLORS = params["colors"]
        SHAPES = params["shapes"]

        for sprite in sprites:
            sprite.move(params["motion"])
            sprite._color = tuple(COLORS[SHAPES.index(sprite._shape) % len(COLORS)])

    elif rule == "MInter":
        SHAPES = params["shapes"]
        COLORS = params["colors"]
        SIZES = params["sizes"]

        colors = []
        shapes = []

        for sprite in sprites:
            shapes.append(SHAPES.index(sprite._shape))
            colors.append(COLORS.index(list(sprite._color)))

        shapes = np.roll(shapes, 1)
        sizes = np.roll(colors, 1)

        for i in range(N):
            sprites[i]._shape = SHAPES[shapes[i]]
            sprites[i]._scale = SIZES[sizes[i]]

            sprites[i]._reset_centered_path()

    elif rule == "MMix1":
        SHAPES = params["shapes"]

        shapes = []

        for sprite in sprites:
            shapes.append(SHAPES.index(sprite._shape))

        shapes = np.roll(shapes, 1)

        for i in range(N):
            sprites[i].move(params["motion"])
            sprites[i]._shape = SHAPES[shapes[i]]

            sprites[i]._reset_centered_path()

    elif rule == "MMix2": 
        COLORS = params["colors"]
        SIZES = params["sizes"]
        SHAPES = params["shapes"]

        colors = []

        for sprite in sprites:
            colors.append(COLORS.index(list(sprite._color)))
            sprite._color = tuple(COLORS[SHAPES.index(sprite._shape) % len(COLORS)])

        sizes = np.roll(colors, 1)

        for i in range(N):
            sprites[i]._scale = SIZES[sizes[i]]

            sprites[i]._reset_centered_path()

    elif rule == "NCP":

        SIZES = params["sizes"]
        sizes = []

        for sprite in sprites:
            sizes.append(SIZES.index(sprite._scale))

        sizes = np.roll(sizes, 1)

        for i in range(N):

            if sizes[i] == 0:
                mov_dir = (params["motion"][1], -1 * params["motion"][0])
            elif sizes[i] == 1:
                mov_dir = (params["motion"][1], 1 * params["motion"][0])
            elif sizes[i] == 2:
                mov_dir = (-1 * params["motion"][0], params["motion"][1])
            elif sizes[i] == 3:
                mov_dir = (1 * params["motion"][0], params["motion"][1])

            sprites[i].move(mov_dir)

    elif rule == "NShAC":

        COLORS = params["colors"]
        SHAPES = params["shapes"]

        for sprite in sprites:

            x = sprite._position[0]
            y = sprite._position[1]

            if x < 0.5 and y < 0.5:
                quad = 3
            elif x >= 0.5 and y < 0.5:
                quad = 4
            elif x < 0.5 and y >= 0.5:
                quad = 2
            else:
                quad = 1

            sprite._color = tuple(COLORS[(SHAPES.index(sprite._shape) + quad) % len(COLORS)])

    elif rule == "NShAASh":

        SHAPES = params["shapes"]
        shapes = []

        for sprite in sprites:
            shapes.append(SHAPES.index(sprite._shape))

        shapes = np.roll(shapes, 1)

        for i in range(N):
            sprites[i]._shape = SHAPES[(SHAPES.index(sprites[i]._shape) + shapes[i]) % len(SHAPES)]
            sprites[i]._reset_centered_path()

    elif rule == "NCAASi":
        COLORS = params["colors"]
        SIZES = params["sizes"]

        colors = []

        for sprite in sprites:

            x = sprite._position[0]
            y = sprite._position[1]

            if x < 0.5 and y < 0.5:
                quad = 3
            elif x >= 0.5 and y < 0.5:
                quad = 4
            elif x < 0.5 and y >= 0.5:
                quad = 2
            else:
                quad = 1

            colors.append((COLORS.index(list(sprite._color)) + quad) % len(SIZES))

        sizes = np.roll(colors, 1)

        for i in range(N):
            sprites[i]._scale = SIZES[sizes[i]]
            sprites[i]._reset_centered_path()

    elif rule == "MNIntra":

        SIZES = params["sizes"]
        COLORS = params["colors"]
        SHAPES = params["shapes"]
        sizes = []

        for sprite in sprites:
            sizes.append(SIZES.index(sprite._scale))
            
            x = sprite._position[0]
            y = sprite._position[1]

            if x < 0.5 and y < 0.5:
                quad = 3
            elif x >= 0.5 and y < 0.5:
                quad = 4
            elif x < 0.5 and y >= 0.5:
                quad = 2
            else:
                quad = 1

            sprite._color = tuple(COLORS[(SHAPES.index(sprite._shape) + quad) % len(COLORS)])

        sizes = np.roll(sizes, 1)

        for i in range(N):

            if sizes[i] == 0:
                mov_dir = (params["motion"][1], -1 * params["motion"][0])
            elif sizes[i] == 1:
                mov_dir = (params["motion"][1], 1 * params["motion"][0])
            elif sizes[i] == 2:
                mov_dir = (-1 * params["motion"][0], params["motion"][1])
            elif sizes[i] == 3:
                mov_dir = (1 * params["motion"][0], params["motion"][1])

            sprites[i].move(mov_dir)

    elif rule == "MNInter":

        SHAPES = params["shapes"]
        COLORS = params["colors"]
        SIZES = params["sizes"]
        shapes = []
        colors = []

        for sprite in sprites:
            shapes.append(SHAPES.index(sprite._shape))
            
            x = sprite._position[0]
            y = sprite._position[1]

            if x < 0.5 and y < 0.5:
                quad = 3
            elif x >= 0.5 and y < 0.5:
                quad = 4
            elif x < 0.5 and y >= 0.5:
                quad = 2
            else:
                quad = 1

            colors.append((COLORS.index(list(sprite._color)) + quad) % len(SIZES))

        shapes = np.roll(shapes, 1)
        sizes = np.roll(colors, 1)

        for i in range(N):
            sprites[i]._shape = SHAPES[(SHAPES.index(sprites[i]._shape) + shapes[i]) % len(SHAPES)] 
            sprites[i]._scale = SIZES[sizes[i]]
            sprites[i]._reset_centered_path()

    elif rule == "MNMix1":

        SIZES = params["sizes"]
        SHAPES = params["shapes"]
        shapes = []
        sizes = []

        for sprite in sprites:
            sizes.append(SIZES.index(sprite._scale))
            shapes.append(SHAPES.index(sprite._shape))

        shapes = np.roll(shapes, 1)
        sizes = np.roll(sizes, 1)

        for i in range(N):

            if sizes[i] == 0:
                mov_dir = (params["motion"][1], -1 * params["motion"][0])
            elif sizes[i] == 1:
                mov_dir = (params["motion"][1], 1 * params["motion"][0])
            elif sizes[i] == 2:
                mov_dir = (-1 * params["motion"][0], params["motion"][1])
            elif sizes[i] == 3:
                mov_dir = (1 * params["motion"][0], params["motion"][1])

            sprites[i].move(mov_dir)

            sprites[i]._shape = SHAPES[(SHAPES.index(sprites[i]._shape) + shapes[i]) % len(SHAPES)]
            sprites[i]._reset_centered_path()

    elif rule == "MNMix2":
        COLORS = params["colors"]
        SHAPES = params["shapes"]
        SIZES = params["sizes"]

        colors = []

        for sprite in sprites:

            x = sprite._position[0]
            y = sprite._position[1]

            if x < 0.5 and y < 0.5:
                quad = 3
            elif x >= 0.5 and y < 0.5:
                quad = 4
            elif x < 0.5 and y >= 0.5:
                quad = 2
            else:
                quad = 1

            colors.append((COLORS.index(list(sprite._color)) + quad) % len(SIZES))

        sizes = np.roll(colors, 1)

        for i in range(N):
            sprite = sprites[i]

            x = sprite._position[0]
            y = sprite._position[1]

            if x < 0.5 and y < 0.5:
                quad = 3
            elif x >= 0.5 and y < 0.5:
                quad = 4
            elif x < 0.5 and y >= 0.5:
                quad = 2
            else:
                quad = 1

            sprite._color = tuple(COLORS[(SHAPES.index(sprite._shape) + quad) % len(COLORS)])
            sprite._scale = SIZES[sizes[i]]
            sprite._reset_centered_path()


    elif rule == "MOOD":

        SHAPES = params["shapes"]
        COLORS = params["colors"]
        SIZES = params["sizes"]

        colors = []
        shapes = []

        for sprite in sprites:
            shapes.append(SHAPES.index(sprite._shape))
            colors.append(COLORS.index(list(sprite._color)))

            sprite._color = tuple(COLORS[SHAPES.index(sprite._shape) % len(COLORS)])

        shapes = np.roll(shapes, 1)
        sizes = np.roll(colors, 1)

        for i in range(N):
            sprites[i]._shape = SHAPES[shapes[i]]
            sprites[i]._scale = SIZES[sizes[i]]

            sprites[i]._reset_centered_path()


    elif rule == "MFull":

        SHAPES = params["shapes"]
        COLORS = params["colors"]
        SIZES = params["sizes"]

        colors = []
        shapes = []

        for sprite in sprites:
            shapes.append(SHAPES.index(sprite._shape))
            colors.append(COLORS.index(list(sprite._color)))

            sprite._color = tuple(COLORS[SHAPES.index(sprite._shape) % len(COLORS)])

        shapes = np.roll(shapes, 1)
        sizes = np.roll(colors, 1)

        for i in range(N):
            sprites[i]._shape = SHAPES[shapes[i]]
            sprites[i]._scale = SIZES[sizes[i]]
            sprites[i].move(params["motion"])

            sprites[i]._reset_centered_path()

    elif rule == "NA":

        COLORS = params["colors"]
        SHAPES = params["shapes"]
        SIZES = params["sizes"]

        sizes = []

        for sprite in sprites:
            sizes.append(SIZES.index(sprite._scale))

        sizes = np.roll(sizes, 1)

        for i, sprite in enumerate(sprites):
            sprite._color = tuple(COLORS[(sizes[i] +  SHAPES.index(sprite._shape)) % len(COLORS)])

    elif rule == "poschangecolor":
        COLORS = params["colors"]

        for sprite in sprites:
            x = sprite._position[0]
            y = sprite._position[1]

            if x < 0.5 and y < 0.5:
                sprite._color = tuple(COLORS[3 % len(COLORS)])
            elif x >= 0.5 and y < 0.5:
                sprite._color = tuple(COLORS[0 % len(COLORS)])
            elif x < 0.5 and y >= 0.5:
                sprite._color = tuple(COLORS[2 % len(COLORS)])
            else:
                sprite._color = tuple(COLORS[1 % len(COLORS)])

    elif rule == "colorwithshape":
        COLORS = params["colors"]
        SHAPES = params["shapes"]

        for sprite in sprites:
            sprite._color = tuple(COLORS[(COLORS.index(list(sprite._color)) +  SHAPES.index(sprite._shape)) % len(COLORS)])


    elif rule == "colorwithposition":
        COLORS = params["colors"]
        
        for sprite in sprites:
            x = sprite._position[0]
            y = sprite._position[1]

            if x < 0.5 and y < 0.5:
                sprite._color = tuple(COLORS[(COLORS.index(list(sprite._color)) +  3) % len(COLORS)])
            elif x >= 0.5 and y < 0.5:
                sprite._color = tuple(COLORS[(COLORS.index(list(sprite._color)) +  4) % len(COLORS)])
            elif x < 0.5 and y >= 0.5:
                sprite._color = tuple(COLORS[(COLORS.index(list(sprite._color)) +  2) % len(COLORS)])
            else:
                sprite._color = tuple(COLORS[(COLORS.index(list(sprite._color)) +  1) % len(COLORS)])


    return sprites


def generate_gt(sprites):
    objects = []

    for i, sprite in enumerate(sprites):
        objects.append({
            'shape': sprite._shape,
            'size': sprite._scale,
            'rotation': sprite._angle,
            '2d_coords': list(sprite._position),
            'color': sprite._color,
            'depth': i,
            })
    return objects

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

def sort_and_render(renderer, sprites, COLORS):
    sorted_sprites = sorted(sprites, key=lambda sprite: COLORS.index(list(sprite._color)))
    return renderer.render(sorted_sprites)


if __name__ == '__main__':
    ''' Unit testing'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--save_path', default='data/')

    parser.add_argument('--num_train', type=int, default=64000)
    parser.add_argument('--num_test', type=int, default=12800)

    parser.add_argument('--train_ratio', type=float, default=0.0)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    parser.add_argument('--rule', default='CP')

    parser.add_argument('--pretrain', default=False, action='store_true')

    args = parser.parse_args()
    
    random.seed(0)
    np.random.seed(0)

    # PARAMETERS
    C, H, W = 3, args.image_size, args.image_size

    run_num = args.num_train
    run_num_test = args.num_test

    # Attribute Define
    SHAPES = ['circle', 'triangle', 'square', 'star_4']
    COLORS = [[0, 255, 0], [255, 0, 255], [0, 127, 255], [255, 127, 0]]
    SIZES = [0.125, 0.225, 0.325, 0.425]

    
    # Attribute Generation
    N_min, N_max = 2, 2

    '''
    Att_N = len(SHAPES)
    size_min, size_max = 0.1, 0.6

    COLORS = generate_colors(Att_N)
    SIZES = generate_sizes(Att_N, size_min, size_max)
    '''

    # Composition space ratio
    test_ratio = args.test_ratio
    train_ratio = args.train_ratio
    assert train_ratio <= 1-test_ratio, "train/test split ratio"

    # PARAMETERS per TASK
    R = args.rule
    if R not in RULES:
        print("Undefined Rule")
        exit()

    params = {
            "motion": (0.3, 0),
            "colors": COLORS,
            "sizes": SIZES,
            "shapes": SHAPES,
            }

    # Binding Pairs (TRAIN)
    BINDS, CORES = generate_binds(COLORS, SHAPES, SIZES, ratio = train_ratio)               # train
    UNSEEN_BINDS = generate_unseen_binds(COLORS, SHAPES, SIZES, BINDS, ratio = test_ratio)  # test

    print(f"Train Comp: {len(BINDS)}, Test Comp: {len(UNSEEN_BINDS)}")

    # Renderer
    renderer = spriteworld_renderers.PILRenderer(
            image_size=(H, W),
        anti_aliasing=10,
        bg_color='black'
    )

    save_path = os.path.join(args.save_path, f"dsprites-{R}-alpha-{train_ratio}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_path = os.path.join(save_path, f"train")
    test_path = os.path.join(save_path, f"test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    train_info = {
            "colors": COLORS,
            "shapes": SHAPES,
            "sizes": SIZES,
            "ratio": train_ratio,
            "binds": BINDS.tolist(),
            "cores": CORES.tolist(),
            }

    test_info = {
            "colors": COLORS,
            "shapes": SHAPES,
            "sizes": SIZES,
            "ratio": test_ratio,
            "unseen_binds": UNSEEN_BINDS.tolist(),
            "cores": CORES.tolist(),
            }

    print(train_info)
    print(test_info)

    with open(os.path.join(train_path, "info.json"), 'w') as outfile:
        json.dump(train_info, outfile)
    with open(os.path.join(test_path, "info.json"), 'w') as outfile:
        json.dump(test_info, outfile)


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

        if not args.pretrain:
            # apply rule
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

    if not args.pretrain: 
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
