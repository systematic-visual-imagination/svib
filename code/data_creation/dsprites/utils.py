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

def sort_and_render(renderer, pairs, COLORS):

    N = len(pairs)

    sorted_pairs = sorted(pairs, key=lambda pair: (COLORS.index(list(pair[0]._color)), pair[0].x))
    
    sorted_unmasked = [pair[0] for pair in sorted_pairs]
    sorted_mask = [pair[1] for pair in sorted_pairs]
    
    img = renderer.render(sorted_unmasked)

    old_colors = []

    for i in range(N):
        old_colors.append(sorted_unmasked[i]._color)
        sorted_unmasked[i]._color = sorted_mask[i]

    mask = renderer.render(sorted_unmasked)

    for sprite, old_color in zip(sorted_unmasked, old_colors):
        sprite._color = old_color

    return img, mask

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

def generate_binds(colors_, shapes_, sizes_, binds_, ratio = 0.8):
    # if binds_ is empty, cores return core combinations.
    # if not, cores return binds_ itself.

    must = max([len(colors_), len(shapes_), len(sizes_)])

    N = len(colors_) * len(shapes_) * len(sizes_)
    N_sample = math.floor((N - must) * ratio)
    
    N_binds_ = len(binds_)

    if N_binds_ == 0:
        # if binds_ is empty, generate binds with cores.
        
        colors = np.append(np.arange(len(colors_)), np.random.randint(low = len(colors_), size = must - len(colors_)))
        shapes = np.append(np.arange(len(shapes_)), np.random.randint(low = len(shapes_), size = must - len(shapes_)))
        sizes = np.append(np.arange(len(sizes_)), np.random.randint(low = len(sizes_), size = must - len(sizes_)))

        color = np.random.choice(colors, (must,1), replace=False)
        shape = np.random.choice(shapes, (must,1), replace=False)
        size = np.random.choice(sizes, (must,1), replace=False)

        binds = np.concatenate((color, shape, size), axis=-1)

    else:
        # binds_ is not empty.
        # check if binds_ contain all cores.
        N_sample = N_sample - (N_binds_ - must)
        assert N_sample >= 0

        binds = binds_

    cores = binds[:]

    pool = np.array(np.meshgrid(np.arange(len(colors_)), np.arange(len(shapes_)), np.arange(len(sizes_)))).T.reshape(-1,3)
    pool = np.array(list(set(map(tuple, pool)) - set(map(tuple, cores))))
   
    for i in range(N_sample):
        idx = np.random.choice(np.arange(pool.shape[0]))
        sample = pool[idx]
        pool = np.delete(pool, idx, axis=0)
        binds = np.concatenate([binds, [sample]], axis=0)

    return binds, cores

def generate_unseen_binds(colors_, shapes_, sizes_, binds_, ratio = 0.2):

    colors = np.arange(len(colors_))
    shapes = np.arange(len(shapes_))
    sizes = np.arange(len(sizes_))

    must = max([len(colors_), len(shapes_), len(sizes_)])

    N = len(colors_)*len(shapes_)*len(sizes_)
    N_sample = N - math.floor((N - must) * (1 - ratio)) - must

    if N_sample > N - len(binds_): # switch to i.i.d
        print("switch to i.i.d")
        return binds_[:]

    unseen_binds_ = []

    unseen_binds_ = np.array(np.meshgrid(colors, shapes, sizes)).T.reshape(-1,3)
    binds_ = np.array(binds_)

    rows1 = unseen_binds_.view([('', unseen_binds_.dtype)] * unseen_binds_.shape[1])
    rows2 = binds_.view([('', binds_.dtype)] * binds_.shape[1])
    unseen_binds_ = np.setdiff1d(rows1, rows2).view(unseen_binds_.dtype).reshape(-1, unseen_binds_.shape[1])
    
    binds = []

    for i in range(N_sample):
        idx = np.random.choice(np.arange(unseen_binds_.shape[0]))
        sample = unseen_binds_[idx]
        unseen_binds_ = np.delete(unseen_binds_, idx, axis=0)
        binds += [sample]

    return np.array(binds)

def generate_binds_with(colors_, shapes_, sizes_, incl_binds, excl_binds, ratio = 0.8):

    must = max([len(colors_), len(shapes_), len(sizes_)])

    colors = np.arange(len(colors_))
    shapes = np.arange(len(shapes_))
    sizes = np.arange(len(sizes_))

    N = len(colors_)*len(shapes_)*len(sizes_)
    N_sample = math.floor((N - must) * ratio) - (len(incl_binds) - must)

    binds_ = np.concatenate([excl_binds, incl_binds], axis=0)

    if N_sample > N - len(binds_): # switch to i.i.d
        print("switch to i.i.d")
        return binds_[:]

    unseen_binds_ = []

    unseen_binds_ = np.array(np.meshgrid(colors, shapes, sizes)).T.reshape(-1,3)

    rows1 = unseen_binds_.view([('', unseen_binds_.dtype)] * unseen_binds_.shape[1])
    rows2 = binds_.view([('', binds_.dtype)] * binds_.shape[1])
    unseen_binds_ = np.setdiff1d(rows1, rows2).view(unseen_binds_.dtype).reshape(-1, unseen_binds_.shape[1])

    binds = incl_binds[:]

    for i in range(N_sample):
        idx = np.random.choice(np.arange(unseen_binds_.shape[0]))
        sample = unseen_binds_[idx]
        unseen_binds_ = np.delete(unseen_binds_, idx, axis=0)
        binds += [sample]

    return np.array(binds)

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
