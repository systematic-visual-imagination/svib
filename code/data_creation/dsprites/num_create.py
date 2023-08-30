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

# Constant
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)

RULES = ["numswapcolor", "numsumcolor", "numchangecolor"]

def generate_colors(Att_N):
    split_n = math.floor((Att_N + 2)**(1/3))

    n1 = split_n**3 - 2
    n2 = Att_N - n1 # Att_N = n1 + n2
    
    if split_n > 1:
        split_c = 1 / (split_n - 1)
        axis_c = [i * split_c for i in range(split_n)]
        colors = np.array(np.meshgrid(axis_c, axis_c, axis_c)).T.reshape(-1,3).tolist()[1:-1]
        colors += distinctipy.get_colors(n2, exclude_colors = colors + [WHITE, BLACK], n_attempts=0)
    
    else:
        colors = distinctipy.get_colors(Att_N, n_attempts = 0)
    
    colors = (np.array(colors)*255).astype(int).tolist()

    return colors

def generate_sizes(Att_N, size_min, size_max):
    split_s = (size_max - size_min)/Att_N
    sizes = [size_min + i*split_s for i in range(Att_N)]

    return sizes

def apply_rule(sprites, rule, params):
    N = len(sprites)

    if rule == "numswapcolor":
        COLORS = params["colors"]

        colors = []

        for sprite in sprites:
            colors.append(COLORS.index(list(sprite._color)))

        max_c = np.max(colors)
        min_c = np.min(colors)
        max_idx = np.where(colors == max_c)[0]
        min_idx = np.where(colors == min_c)[0]


        colors = np.array(colors)
        np.put(colors, max_idx, min_c)
        np.put(colors, min_idx, max_c)

        for i in range(N):
            sprites[i]._color = tuple(COLORS[colors[i]])

    elif rule == "numsumcolor":
        COLORS = params["colors"]
        N_C = len(COLORS)

        colors = []

        for sprite in sprites:
            colors.append(COLORS.index(list(sprite._color)))

        total = sum(colors)
        
        
        for i in range(N):
            sprites[i]._color = tuple(COLORS[(total+N_C-colors[i]) % N_C])

    elif rule == "numchangecolor":
        COLORS = params["colors"]
        colors = np.array([(COLORS.index(list(sprite._color)) + N) % len(COLORS) for sprite in sprites])
        for i in range(N):
            sprites[i]._color = tuple(COLORS[colors[i]])

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

if __name__ == '__main__':
    ''' Unit testing'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--save_path', default='data/')

    parser.add_argument('--num_train', type=int, default=64000)

    parser.add_argument('--rule', type=int, default=0)
    parser.add_argument('--nmin', type=int, default=2)
    parser.add_argument('--nmax', type=int, default=2)
    args = parser.parse_args()
    
    random.seed(0)
    np.random.seed(0)

    # PARAMETERS
    C, H, W = 3, args.image_size, args.image_size

    run_num = args.num_train

    # Attribute Generation
    SHAPES = ['circle', 'triangle', 'square', 'star_4']
    
    N_min, N_max = args.nmin, args.nmax
    Att_N = len(SHAPES)

    COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0]]
    SIZES = [0.225]

    # PARAMETERS per TASK
    R = RULES[args.rule]
    params = {
            "motion": (0.3, 0),
            "colors": COLORS,
            "sizes": SIZES,
            "shapes": SHAPES,
            }
    
    # Renderer
    renderer = spriteworld_renderers.PILRenderer(
            image_size=(H, W),
        anti_aliasing=10,
        bg_color='black'
    )

    save_path = os.path.join(args.save_path, f"dsprites-{R}-nmin-{N_min}-nmax-{N_max}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_path = os.path.join(save_path, f"train")
    os.makedirs(train_path, exist_ok=True)

    train_info = {
            "colors": COLORS,
            "shapes": SHAPES,
            "sizes": SIZES,
            "N_max": N_max,
            }

    print(train_info)

    with open(os.path.join(train_path, "info.json"), 'w') as outfile:
        json.dump(train_info, outfile)

    print("Train Set : {}".format(run_num))

    for run in tqdm(range(run_num)):
        sample_path = os.path.join(train_path, f"{run:08d}")
        os.makedirs(sample_path, exist_ok=True)

        N = np.random.choice(np.arange(N_min, N_max + 1))
        size = np.random.choice(SIZES, size=(N, 1)) 

        while True:
            x = np.random.random(size=(N, 2)) * (1 - 2 * 0.4 * size) + 0.4 * size
            if checker(x, size * 0.4):
                break


        angle = np.random.random(size=(N)) * 360
        shape = np.random.choice(np.arange(len(SHAPES)), size=(N))
        color = np.random.choice(np.arange(len(COLORS)), size=(N))
        
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


        # apply rule
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
