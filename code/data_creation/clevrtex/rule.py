import numpy as np
from macros import *

def apply_rule(object_binds, x, N, rule):
        
    # angle
    angle = np.zeros(N)

    s_obj_positions, t_obj_positions = [], []
    s_obj_rotations, t_obj_rotations = [], []
    s_obj_shapes, t_obj_shapes = [], []
    s_obj_sizes, t_obj_sizes = [], []
    s_obj_materials, t_obj_materials = [], []

    for i in range(N):
        if rule == 'none':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_shapes.append(SHAPES[object_binds[i][0]])
            s_obj_sizes.append(SIZES[object_binds[i][1]])
            s_obj_materials.append(MATERIALS[object_binds[i][2]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_shapes.append(SHAPES[object_binds[i][0]])
            t_obj_sizes.append(SIZES[object_binds[i][1]])
            t_obj_materials.append(MATERIALS[object_binds[i][2]])

        elif rule == 'CP':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_shapes.append(SHAPES[object_binds[i][0]])
            s_obj_sizes.append(SIZES[object_binds[i][1]])
            s_obj_materials.append(MATERIALS[object_binds[i][2]])

            t_obj_positions.append((np.clip(x[i, 0] + SHIFT_AMOUNT, -3, 3), x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_shapes.append(SHAPES[object_binds[i][0]])
            t_obj_sizes.append(SIZES[object_binds[i][1]])
            t_obj_materials.append(MATERIALS[object_binds[i][2]])

        elif rule == 'ShAASh':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_shapes.append(SHAPES[object_binds[i][0]])
            s_obj_sizes.append(SIZES[object_binds[i][1]])
            s_obj_materials.append(MATERIALS[object_binds[i][2]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_shapes.append(SHAPES[object_binds[(i + 1) % N][0]])
            t_obj_sizes.append(SIZES[object_binds[i][1]])
            t_obj_materials.append(MATERIALS[object_binds[i][2]])

        elif rule == 'MAASi':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_shapes.append(SHAPES[object_binds[i][0]])
            s_obj_sizes.append(SIZES[object_binds[i][1]])
            s_obj_materials.append(MATERIALS[object_binds[i][2]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_shapes.append(SHAPES[object_binds[i][0]])
            t_obj_sizes.append(SIZES[object_binds[(i + 1) % N][2] % len(SIZES)])
            t_obj_materials.append(MATERIALS[object_binds[i][2]])

        elif rule == 'NMAASi':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_shapes.append(SHAPES[object_binds[i][0]])
            s_obj_sizes.append(SIZES[object_binds[i][1]])
            s_obj_materials.append(MATERIALS[object_binds[i][2]])

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
            t_obj_shapes.append(SHAPES[object_binds[i][0]])
            t_obj_sizes.append(SIZES[(object_binds[(i + 1) % N][2] + quad) % len(SIZES)])
            t_obj_materials.append(MATERIALS[object_binds[i][2]])

        elif rule == 'ShAM+MAASi':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_shapes.append(SHAPES[object_binds[i][0]])
            s_obj_sizes.append(SIZES[object_binds[i][1]])
            s_obj_materials.append(MATERIALS[object_binds[i][2]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_shapes.append(SHAPES[object_binds[i][0]])
            t_obj_sizes.append(SIZES[object_binds[(i + 1) % N][2] % len(SIZES)])
            t_obj_materials.append(MATERIALS[object_binds[i][0] % len(MATERIALS)])

        elif rule == 'NShAASh':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_shapes.append(SHAPES[object_binds[i][0]])
            s_obj_sizes.append(SIZES[object_binds[i][1]])
            s_obj_materials.append(MATERIALS[object_binds[i][2]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_shapes.append(SHAPES[(object_binds[(i + 1) % N][0] + object_binds[i][0]) % len(SHAPES)])
            t_obj_sizes.append(SIZES[object_binds[i][1]])
            t_obj_materials.append(MATERIALS[object_binds[i][2]])

        elif rule == 'ShAASh+MAASi':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_shapes.append(SHAPES[object_binds[i][0]])
            s_obj_sizes.append(SIZES[object_binds[i][1]])
            s_obj_materials.append(MATERIALS[object_binds[i][2]])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_shapes.append(SHAPES[object_binds[(i + 1) % N][0]])
            t_obj_sizes.append(SIZES[object_binds[(i + 1) % N][2] % len(SIZES)])
            t_obj_materials.append(MATERIALS[object_binds[i][2]])

        elif rule == 'NShAASh+NMAASi':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_shapes.append(SHAPES[object_binds[i][0]])
            s_obj_sizes.append(SIZES[object_binds[i][1]])
            s_obj_materials.append(MATERIALS[object_binds[i][2]])

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
            t_obj_shapes.append(SHAPES[(object_binds[(i + 1) % N][0] + object_binds[i][0]) % len(SHAPES)])
            t_obj_sizes.append(SIZES[(object_binds[(i + 1) % N][2] + quad) % len(SIZES)])
            t_obj_materials.append(MATERIALS[object_binds[i][2]])

        elif rule == 'NShAM+NMAASi':
            s_obj_positions.append((x[i, 0], x[i, 1]))
            s_obj_rotations.append(angle[i])
            s_obj_shapes.append(SHAPES[object_binds[i][0]])
            s_obj_sizes.append(SIZES[object_binds[i][1]])
            s_obj_materials.append(MATERIALS[object_binds[i][2]])

            def get_quad(_x, _y):
                if _x < 0.0 and _y < 0.0:
                    quad = 3
                elif _x >= 0.0 and _y < 0.0:
                    quad = 4
                elif _x < 0.0 and _y >= 0.0:
                    quad = 2
                else:
                    quad = 1
                return quad

            self_quad = get_quad(x[i, 0], x[i, 1])
            other_quad = get_quad(x[(i + 1) % N, 0], x[(i + 1) % N, 1])

            t_obj_positions.append((x[i, 0], x[i, 1]))
            t_obj_rotations.append(angle[i])
            t_obj_shapes.append(SHAPES[object_binds[i][0]])
            t_obj_sizes.append(SIZES[(object_binds[(i + 1) % N][2] + other_quad) % len(SIZES)])
            t_obj_materials.append(MATERIALS[(object_binds[i][0] + self_quad) % len(MATERIALS)])

        else:
            raise NotImplementedError

    return s_obj_positions, s_obj_rotations, s_obj_shapes, s_obj_sizes, s_obj_materials, \
            t_obj_positions, t_obj_rotations, t_obj_shapes, t_obj_sizes, t_obj_materials
