import numpy as np
from spriteworld.sprite import Sprite

from macros import *

RULES = [
        "None",
        "CP", "ShAC", "ShAASh", "CAASi",
        "MIntra", "MMix2", "MMix1", "MInter",
        "NCP", "NShAC", "NShAASh", "NCAASi",
        "MNIntra", "MNMix2", "MNMix1", "MNInter",
        ]

def apply_rule(sprites, rule):
    N = len(sprites)

    if rule == "CP":
        for i in range(N):
            sprites[i].move(MOTION)

    elif rule == "ShAC":

        for sprite in sprites:
            sprite._color = tuple(COLORS[SHAPES.index(sprite._shape) % len(COLORS)])

    elif rule == "ShAASh":

        shapes = []

        for sprite in sprites:
            shapes.append(SHAPES.index(sprite._shape))

        shapes = np.roll(shapes, 1)

        for i in range(N):
            sprites[i]._shape = SHAPES[shapes[i]]

            sprites[i]._reset_centered_path()

    elif rule == "CAASi":

        colors = []

        for sprite in sprites:
            colors.append(COLORS.index(list(sprite._color)))

        sizes = np.roll(colors, 1)

        for i in range(N):
            sprites[i]._scale = SIZES[sizes[i]]

            sprites[i]._reset_centered_path()
    

    elif rule == "MIntra":
        # CP & ShAC

        for sprite in sprites:
            sprite.move(MOTION)
            sprite._color = tuple(COLORS[SHAPES.index(sprite._shape) % len(COLORS)])
    
    elif rule == "MMix2": 

        colors = []

        for sprite in sprites:
            colors.append(COLORS.index(list(sprite._color)))
            sprite._color = tuple(COLORS[SHAPES.index(sprite._shape) % len(COLORS)])

        sizes = np.roll(colors, 1)

        for i in range(N):
            sprites[i]._scale = SIZES[sizes[i]]

            sprites[i]._reset_centered_path()

    elif rule == "MMix1":

        shapes = []

        for sprite in sprites:
            shapes.append(SHAPES.index(sprite._shape))

        shapes = np.roll(shapes, 1)

        for i in range(N):
            sprites[i].move(MOTION)
            sprites[i]._shape = SHAPES[shapes[i]]

            sprites[i]._reset_centered_path()

    elif rule == "MInter":

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


    elif rule == "NCP":

        sizes = []

        for sprite in sprites:
            sizes.append(SIZES.index(sprite._scale))

        sizes = np.roll(sizes, 1)

        for i in range(N):

            if sizes[i] == 0:
                mov_dir = (MOTION[1], -1 * MOTION[0])
            elif sizes[i] == 1:
                mov_dir = (MOTION[1], 1 * MOTION[0])
            elif sizes[i] == 2:
                mov_dir = (-1 * MOTION[0], MOTION[1])
            elif sizes[i] == 3:
                mov_dir = (1 * MOTION[0], MOTION[1])

            sprites[i].move(mov_dir)

    elif rule == "NShAC":

        shapes = []

        for sprite in sprites:
            shapes.append(SHAPES.index(sprite._shape))

        shapes = np.roll(shapes, 1)

        for i in range(N):
            sprites[i]._shape = SHAPES[(SHAPES.index(sprites[i]._shape) + shapes[i]) % len(SHAPES)]
            sprites[i]._reset_centered_path()

    elif rule == "NCAASi":

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
                mov_dir = (MOTION[1], -1 * MOTION[0])
            elif sizes[i] == 1:
                mov_dir = (MOTION[1], 1 * MOTION[0])
            elif sizes[i] == 2:
                mov_dir = (-1 * MOTION[0], MOTION[1])
            elif sizes[i] == 3:
                mov_dir = (1 * MOTION[0], MOTION[1])

            sprites[i].move(mov_dir)
    
    elif rule == "MNMix2":

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

    elif rule == "MNMix1":

        shapes = []
        sizes = []

        for sprite in sprites:
            sizes.append(SIZES.index(sprite._scale))
            shapes.append(SHAPES.index(sprite._shape))

        shapes = np.roll(shapes, 1)
        sizes = np.roll(sizes, 1)

        for i in range(N):

            if sizes[i] == 0:
                mov_dir = (MOTION[1], -1 * MOTION[0])
            elif sizes[i] == 1:
                mov_dir = (MOTION[1], 1 * MOTION[0])
            elif sizes[i] == 2:
                mov_dir = (-1 * MOTION[0], MOTION[1])
            elif sizes[i] == 3:
                mov_dir = (1 * MOTION[0], MOTION[1])

            sprites[i].move(mov_dir)

            sprites[i]._shape = SHAPES[(SHAPES.index(sprites[i]._shape) + shapes[i]) % len(SHAPES)]
            sprites[i]._reset_centered_path()

    elif rule == "MNInter":

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
    
    return sprites


