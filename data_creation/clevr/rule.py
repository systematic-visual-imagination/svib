import numpy as np
from macros import *

def apply_rule(object_binds, x, N, rule):

    # angle
    angle = np.zeros(N)

    s_obj_positions, t_obj_positions = [], []
    s_obj_rotations, t_obj_rotations = [], []
    s_obj_colors, t_obj_colors = [], []
    s_obj_shapes, t_obj_shapes = [], []
    s_obj_sizes, t_obj_sizes = [], []
    s_obj_materials, t_obj_materials = [], []

    for i in range(N):
        if rule == 'none':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_colors.append(COLORS[object_binds[i][0]])
            s_obj_shapes.append(SHAPES[object_binds[i][1]])
            s_obj_sizes.append(SIZES[object_binds[i][2]])
            s_obj_materials.append(MATERIALS[object_binds[i][3]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_colors.append(COLORS[object_binds[i][0]])
            t_obj_shapes.append(SHAPES[object_binds[i][1]])
            t_obj_sizes.append(SIZES[object_binds[i][2]])
            t_obj_materials.append(MATERIALS[object_binds[i][3]])

        elif rule == 'xshift':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_colors.append(COLORS[object_binds[i][0]])
            s_obj_shapes.append(SHAPES[object_binds[i][1]])
            s_obj_sizes.append(SIZES[object_binds[i][2]])
            s_obj_materials.append(MATERIALS[object_binds[i][3]])

            t_obj_positions.append((np.clip(x[i, 0] + SHIFT_AMOUNT, -3, 3), x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_colors.append(COLORS[object_binds[i][0]])
            t_obj_shapes.append(SHAPES[object_binds[i][1]])
            t_obj_sizes.append(SIZES[object_binds[i][2]])
            t_obj_materials.append(MATERIALS[object_binds[i][3]])

        elif rule == 'swap-shape':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_colors.append(COLORS[object_binds[i][0]])
            s_obj_shapes.append(SHAPES[object_binds[i][1]])
            s_obj_sizes.append(SIZES[object_binds[i][2]])
            s_obj_materials.append(MATERIALS[object_binds[i][3]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_colors.append(COLORS[object_binds[i][0]])
            t_obj_shapes.append(SHAPES[object_binds[(i + 1) % N][1]])
            t_obj_sizes.append(SIZES[object_binds[i][2]])
            t_obj_materials.append(MATERIALS[object_binds[i][3]])

        elif rule == 'shape-affects-color':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_colors.append(COLORS[object_binds[i][0]])
            s_obj_shapes.append(SHAPES[object_binds[i][1]])
            s_obj_sizes.append(SIZES[object_binds[i][2]])
            s_obj_materials.append(MATERIALS[object_binds[i][3]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_colors.append(COLORS[object_binds[i][1] % len(COLORS)])
            t_obj_shapes.append(SHAPES[object_binds[i][1]])
            t_obj_sizes.append(SIZES[object_binds[i][2]])
            t_obj_materials.append(MATERIALS[object_binds[i][3]])

        elif rule == 'change-color':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_colors.append(COLORS[object_binds[i][0]])
            s_obj_shapes.append(SHAPES[object_binds[i][1]])
            s_obj_sizes.append(SIZES[object_binds[i][2]])
            s_obj_materials.append(MATERIALS[object_binds[i][3]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_colors.append(COLORS[(object_binds[i][0] + 1) % len(COLORS)])
            t_obj_shapes.append(SHAPES[object_binds[i][1]])
            t_obj_sizes.append(SIZES[object_binds[i][2]])
            t_obj_materials.append(MATERIALS[object_binds[i][3]])

        elif rule == 'color-affects-another-size':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_colors.append(COLORS[object_binds[i][0]])
            s_obj_shapes.append(SHAPES[object_binds[i][1]])
            s_obj_sizes.append(SIZES[object_binds[i][2]])
            s_obj_materials.append(MATERIALS[object_binds[i][3]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_colors.append(COLORS[object_binds[i][0]])
            t_obj_shapes.append(SHAPES[object_binds[i][1]])
            t_obj_sizes.append(SIZES[object_binds[(i + 1) % N][0] % len(SIZES)])
            t_obj_materials.append(MATERIALS[object_binds[i][3]])

        elif rule == 'NShAASh':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_colors.append(COLORS[object_binds[i][0]])
            s_obj_shapes.append(SHAPES[object_binds[i][1]])
            s_obj_sizes.append(SIZES[object_binds[i][2]])
            s_obj_materials.append(MATERIALS[object_binds[i][3]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_colors.append(COLORS[object_binds[i][0]])
            t_obj_shapes.append(SHAPES[(object_binds[(i + 1) % N][1] + object_binds[i][1]) % len(SHAPES)])
            t_obj_sizes.append(SIZES[object_binds[i][2]])
            t_obj_materials.append(MATERIALS[object_binds[i][3]])

        elif rule == 'NCAASi':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_colors.append(COLORS[object_binds[i][0]])
            s_obj_shapes.append(SHAPES[object_binds[i][1]])
            s_obj_sizes.append(SIZES[object_binds[i][2]])
            s_obj_materials.append(MATERIALS[object_binds[i][3]])

            quad_x = x[(i + 1) % N, 0]
            quad_y = x[(i + 1) % N, 1]

            if quad_x < 0.0 and quad_y < 0.0:
                quad = 3
            elif quad_x >= 0.0 and quad_y < 0.0:
                quad = 4
            elif quad_x < 0.0 and quad_y >= 0.0:
                quad = 2
            else:
                quad = 1

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_colors.append(COLORS[object_binds[i][0]])
            t_obj_shapes.append(SHAPES[object_binds[i][1]])
            t_obj_sizes.append(SIZES[(object_binds[(i + 1) % N][0] + quad) % len(SIZES)])
            t_obj_materials.append(MATERIALS[object_binds[i][3]])

        elif rule == 'M-Mix2':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_colors.append(COLORS[object_binds[i][0]])
            s_obj_shapes.append(SHAPES[object_binds[i][1]])
            s_obj_sizes.append(SIZES[object_binds[i][2]])
            s_obj_materials.append(MATERIALS[object_binds[i][3]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_colors.append(COLORS[object_binds[i][1] % len(COLORS)])
            t_obj_shapes.append(SHAPES[object_binds[i][1]])
            t_obj_sizes.append(SIZES[object_binds[(i + 1) % N][0] % len(SIZES)])
            t_obj_materials.append(MATERIALS[object_binds[i][3]])

        elif rule == 'M-Inter':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_colors.append(COLORS[object_binds[i][0]])
            s_obj_shapes.append(SHAPES[object_binds[i][1]])
            s_obj_sizes.append(SIZES[object_binds[i][2]])
            s_obj_materials.append(MATERIALS[object_binds[i][3]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_colors.append(COLORS[object_binds[i][0]])
            t_obj_shapes.append(SHAPES[object_binds[(i + 1) % N][1]])
            t_obj_sizes.append(SIZES[object_binds[(i + 1) % N][0] % len(SIZES)])
            t_obj_materials.append(MATERIALS[object_binds[i][3]])

        elif rule == 'M-N-Mix2':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_colors.append(COLORS[object_binds[i][0]])
            s_obj_shapes.append(SHAPES[object_binds[i][1]])
            s_obj_sizes.append(SIZES[object_binds[i][2]])
            s_obj_materials.append(MATERIALS[object_binds[i][3]])

            quad_x = x[i, 0]
            quad_y = x[i, 1]

            if quad_x < 0.0 and quad_y < 0.0:
                quad_c = 3
            elif quad_x >= 0.0 and quad_y < 0.0:
                quad_c = 4
            elif quad_x < 0.0 and quad_y >= 0.0:
                quad_c = 2
            else:
                quad_c = 1

            quad_x = x[(i + 1) % N, 0]
            quad_y = x[(i + 1) % N, 1]

            if quad_x < 0.0 and quad_y < 0.0:
                quad_s = 3
            elif quad_x >= 0.0 and quad_y < 0.0:
                quad_s = 4
            elif quad_x < 0.0 and quad_y >= 0.0:
                quad_s = 2
            else:
                quad_s = 1

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_colors.append(COLORS[(object_binds[i][1] + quad_c) % len(COLORS)])
            t_obj_shapes.append(SHAPES[object_binds[i][1]])
            t_obj_sizes.append(SIZES[(object_binds[(i + 1) % N][0] + quad_s) % len(SIZES)])
            t_obj_materials.append(MATERIALS[object_binds[i][3]])
        
        elif rule == 'M-N-Inter':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_colors.append(COLORS[object_binds[i][0]])
            s_obj_shapes.append(SHAPES[object_binds[i][1]])
            s_obj_sizes.append(SIZES[object_binds[i][2]])
            s_obj_materials.append(MATERIALS[object_binds[i][3]])

            quad_x = x[(i + 1) % N, 0]
            quad_y = x[(i + 1) % N, 1]

            if quad_x < 0.0 and quad_y < 0.0:
                quad = 3
            elif quad_x >= 0.0 and quad_y < 0.0:
                quad = 4
            elif quad_x < 0.0 and quad_y >= 0.0:
                quad = 2
            else:
                quad = 1

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_colors.append(COLORS[object_binds[i][0]])
            t_obj_shapes.append(SHAPES[(object_binds[(i + 1) % N][1] + object_binds[i][1]) % len(SHAPES)])
            t_obj_sizes.append(SIZES[(object_binds[(i + 1) % N][0] + quad) % len(SIZES)])
            t_obj_materials.append(MATERIALS[object_binds[i][3]])

        else:
            raise NotImplementedError
    
    return s_obj_positions, s_obj_rotations, s_obj_colors, s_obj_shapes, s_obj_sizes, s_obj_materials, \
            t_obj_positions, t_obj_rotations, t_obj_colors, t_obj_shapes, t_obj_sizes, t_obj_materials
