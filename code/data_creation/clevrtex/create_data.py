from __future__ import print_function

import argparse
import json
import os
import random
import sys
import tempfile
import shutil

import numpy as np

from macros import *
from collections import Counter

"""
This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import utils
    except ImportError as e:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import utils.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/scene.blend',
                    help="Base blender file on which all scenes are based; includes " +
                         "ground plane, lights, and camera.")
parser.add_argument('--shape_dir', default='data/shapes',
                    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
                    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
                    help="Optional path to a JSON file mapping shape names to a list of " +
                         "allowed color names for that shape. This allows rendering images " +
                         "for CLEVR-CoGenT.")


# Rendering options
parser.add_argument('--use_gpu', default=1, type=int,
                    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
                         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
                         "to work.")
parser.add_argument('--width', default=128, type=int,
                    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=128, type=int,
                    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
                    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
                    help="The number of samples to use when rendering. Larger values will " +
                         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
                    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
                    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
                    help="The tile size to use for rendering. This should not affect the " +
                         "quality of the rendered image but may affect the speed; CPU-based " +
                         "rendering may achieve better performance using smaller tile sizes " +
                         "while larger tile sizes may be optimal for GPU-based rendering.")

# Object options
parser.add_argument('--min_objects', default=2, type=int,
                    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=2, type=int,
                    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_pixels_per_object', default=100, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")

# Output options
parser.add_argument('--save_path', default='output/')
parser.add_argument('--bind_path', default='precomputed_binds/binds.json')

parser.add_argument('--num_samples', type=int, default=80)

parser.add_argument('--rule', default='none')

parser.add_argument('--no_target', default=False, action='store_true')

argv = utils.extract_args()
args = parser.parse_args(argv)


def render_scene(num_objects,
                 source_positions, source_rotations, source_shapes, source_sizes, source_materials,
                 target_positions, target_rotations, target_shapes, target_sizes, target_materials,
                 args,
                 output_dir='path/to/dir',
                 target=True
                 ):

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    if args.use_gpu == 1:
        print('Trying to use GPUs!')
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'
        # get_devices() to let Blender detects GPU device
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            d["use"] = 1  # Using all devices, include GPU and CPU
            print(d["name"], d["use"])

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces

    # Add material to ground
    ground = bpy.data.objects['Ground']
    ground_mat = random.choice(MATERIALS)
    ground_mat_blender = utils.add_material(ground, ground_mat)

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

    camera = bpy.data.objects['Camera']

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

    # Now add source objects
    objects, blender_objects, blender_materials = add_objects(num_objects,
                                           source_positions, source_rotations, source_shapes, source_sizes, source_materials,
                                           args, camera)

    all_visible = make_mask_and_check_visibility([ground] + blender_objects, [ground_mat_blender] + blender_materials, args.min_pixels_per_object, os.path.join(output_dir, 'source_mask.png'))
    if not all_visible:
        return False

    # Render the scene and dump the scene data structure
    render_args.filepath = os.path.join(output_dir, 'source.png')
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)

    scene_struct = {
        'image_filename': os.path.basename(output_dir),
        'objects': objects,
        'ground_material': ground_mat,
        'Lamp_Key': list(bpy.data.objects['Lamp_Key'].location),
        'Lamp_Back': list(bpy.data.objects['Lamp_Back'].location),
        'Lamp_Fill': list(bpy.data.objects['Lamp_Fill'].location),
        'Camera': list(bpy.data.objects['Camera'].location),
    }
    with open(os.path.join(output_dir, 'source.json'), 'w') as f:
        json.dump(scene_struct, f, indent=2)

    if target:
        # delete all source objects
        for obj in blender_objects:
            utils.delete_object(obj)

        # now add target objects
        objects, blender_objects, blender_materials = add_objects(num_objects,
                                               target_positions, target_rotations, target_shapes, target_sizes, target_materials,
                                               args, camera)

        all_visible = make_mask_and_check_visibility([ground] + blender_objects,
                                                     [ground_mat_blender] + blender_materials,
                                                     args.min_pixels_per_object,
                                                     os.path.join(output_dir, 'target_mask.png'))
        if not all_visible:
            return False

        # Render the scene and dump the scene data structure
        render_args.filepath = os.path.join(output_dir, 'target.png')
        while True:
            try:
                bpy.ops.render.render(write_still=True)
                break
            except Exception as e:
                print(e)

        scene_struct = {
            'image_filename': os.path.basename(output_dir),
            'objects': objects,
            'ground_material': ground_mat,
            'Lamp_Key': list(bpy.data.objects['Lamp_Key'].location),
            'Lamp_Back': list(bpy.data.objects['Lamp_Back'].location),
            'Lamp_Fill': list(bpy.data.objects['Lamp_Fill'].location),
            'Camera': list(bpy.data.objects['Camera'].location),
        }
        with open(os.path.join(output_dir, 'target.json'), 'w') as f:
            json.dump(scene_struct, f, indent=2)

    return True


def add_objects(num_objects, positions, rotations, shapes, sizes, materials, args, camera):
    """
    Add objects with specific attributes to the current blender scene
    """

    objects = []
    blender_objects = []
    blender_materials = []
    for i in range(num_objects):

        x, y = positions[i]

        theta = rotations[i]

        size = sizes[i]

        shape = shapes[i]

        material = materials[i]

        utils.add_object(args.shape_dir, shape, size, (x, y), theta=theta)

        obj = bpy.context.object
        blender_objects.append(obj)

        mat = utils.add_material(obj, material)
        blender_materials.append(mat)

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            'shape': shape,
            'size': size,
            'material': material,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
        })

    return objects, blender_objects, blender_materials


def make_mask_and_check_visibility(blender_objects, blender_materials, min_pixels_per_object, path):

    render_shadeless(blender_objects, blender_materials, path=path)

    img = bpy.data.images.load(path)
    p = list(img.pixels)

    def euclidean_similarity(A, B):
        return np.exp(-np.linalg.norm(A - B))

    pixel_colors = []
    for i in range(0, len(p), 4):
        pixel_color = np.asarray([p[i], p[i+1], p[i+2]])
        best_similarity = -1
        best_color = None
        for j, mask_color in enumerate(MASK_COLORS):
            similarity = euclidean_similarity(pixel_color, np.asarray(mask_color[:3]))
            if similarity >= best_similarity:
                best_similarity = similarity
                best_color = j
        pixel_colors += [best_color]

    color_count = Counter(pixel_colors)

    for color in [0, 1, 2]:
        if color not in color_count:
            return False
        else:
            if color_count[color] < min_pixels_per_object:
                return False

    return True


def render_shadeless(blender_objects, blender_materials, path='flat.png'):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Override some render settings to have flat shading
    render_args.filepath = path

    # Add random shadeless materials to all objects
    undo_material_things = []
    for i, (obj, mat) in enumerate(zip(blender_objects, blender_materials)):
        undo_material_things.append(utils.set_to_shadeless(mat, MASK_COLORS[i]))

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Undo the above; first restore the materials to objects
    for thing in undo_material_things:
        utils.undo_shadeless(*thing)


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

    # Load binds
    with open(args.bind_path) as binds_file:
        binds = json.load(binds_file)
        binds = binds["binds"]

    # Dump Set
    save_path = args.save_path
    temp_path = os.path.join(args.save_path, '..', 'TEMP')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)

    done = 0
    while True:
        sample_label = random.randint(0, 10 ** 16)
        sample_dir = "{:018d}".format(sample_label)
        sample_path = os.path.join(temp_path, sample_dir)
        os.makedirs(sample_path, exist_ok=True)

        N = np.random.choice(np.arange(args.min_objects, args.max_objects + 1))
        object_binds = [random.choice(binds) for _ in range(N)]

        x = np.random.uniform(-3, 3, (N, 2))
        while True:
            x = np.random.uniform(-3, 3, (N, 2))
            if checker(x, [SIZES[object_binds[i][1]] for i in range(N)]):
                break

        # angle
        angle = np.zeros(N)

        s_obj_positions, t_obj_positions = [], []
        s_obj_rotations, t_obj_rotations = [], []
        s_obj_shapes, t_obj_shapes = [], []
        s_obj_sizes, t_obj_sizes = [], []
        s_obj_materials, t_obj_materials = [], []

        for i in range(N):
            if args.rule == 'none':
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

            elif args.rule == 'CP':
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

            elif args.rule == 'ShAASh':
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

            elif args.rule == 'MAASi':
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

            elif args.rule == 'NMAASi':
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

            elif args.rule == 'ShAM+MAASi':
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

            elif args.rule == 'NShAASh':
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

            elif args.rule == 'ShAASh+MAASi':
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

            elif args.rule == 'NShAASh+NMAASi':
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

            elif args.rule == 'NShAM+NMAASi':
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

        if not checker(np.asarray(s_obj_positions), s_obj_sizes):
            success = False
        elif not checker(np.asarray(t_obj_positions), t_obj_sizes):
            success = False
        elif not render_scene(N,
                     s_obj_positions, s_obj_rotations, s_obj_shapes, s_obj_sizes, s_obj_materials,
                     t_obj_positions, t_obj_rotations, t_obj_shapes, t_obj_sizes, t_obj_materials,
                     args=args,
                     output_dir=sample_path,
                     target=not args.no_target):
            success = False
        else:
            success = True

        if not success:
            shutil.rmtree(sample_path, ignore_errors=True)
        else:
            shutil.move(sample_path, os.path.join(save_path, sample_dir))
            done += 1

        if done >= args.num_samples:
            break

    print('Dataset saved at : {}'.format(save_path))
