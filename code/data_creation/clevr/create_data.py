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
from rule import apply_rule

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
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
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
                 source_positions, source_rotations, source_colors, source_shapes, source_sizes, source_materials,
                 target_positions, target_rotations, target_colors, target_shapes, target_sizes, target_materials,
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
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        else:
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = 'GPU'

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
    objects, blender_objects = add_objects(num_objects,
                                           source_positions, source_rotations, source_colors, source_shapes, source_sizes, source_materials,
                                           args, camera)

    all_visible = make_mask_and_check_visibility(blender_objects, args.min_pixels_per_object, os.path.join(output_dir, 'source_mask.png'))
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
        objects, blender_objects = add_objects(num_objects,
                                               target_positions, target_rotations, target_colors, target_shapes, target_sizes, target_materials,
                                               args, camera)

        all_visible = make_mask_and_check_visibility(blender_objects, args.min_pixels_per_object, os.path.join(output_dir, 'target_mask.png'))
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
            'Lamp_Key': list(bpy.data.objects['Lamp_Key'].location),
            'Lamp_Back': list(bpy.data.objects['Lamp_Back'].location),
            'Lamp_Fill': list(bpy.data.objects['Lamp_Fill'].location),
            'Camera': list(bpy.data.objects['Camera'].location),
        }
        with open(os.path.join(output_dir, 'target.json'), 'w') as f:
            json.dump(scene_struct, f, indent=2)

    return True


def add_objects(num_objects, positions, rotations, colors, shapes, sizes, materials, args, camera):
    """
    Add objects with specific attributes to the current blender scene
    """

    objects = []
    blender_objects = []
    for i in range(num_objects):

        x, y = positions[i]

        theta = rotations[i]

        size = sizes[i]

        shape = shapes[i]

        material = materials[i]

        color = colors[i]

        utils.add_object(args.shape_dir, shape, size, (x, y), theta=theta)

        obj = bpy.context.object
        blender_objects.append(obj)

        utils.add_material(material, Color=color)

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            'shape': shape,
            'size': size,
            'material': material,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color,
        })

    return objects, blender_objects


def make_mask_and_check_visibility(blender_objects, min_pixels_per_object, path):

    render_shadeless(blender_objects, path=path)

    img = bpy.data.images.load(path)
    p = list(img.pixels)

    color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))

    if len(color_count) != len(blender_objects) + 1:
        return False

    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
          return False

    return True


def render_shadeless(blender_objects, path='flat.png'):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False

    # Move the lights and ground to layer 2 so they don't render
    utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    utils.set_layer(bpy.data.objects['Ground'], 2)

    # Add random shadeless materials to all objects
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        r, g, b = MASK_COLORS[i]
        mat.diffuse_color = [r, g, b]
        mat.use_shadeless = True
        obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    # Move the lights and ground back to layer 0
    utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
    utils.set_layer(bpy.data.objects['Ground'], 0)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing


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
            if checker(x, [SIZES[object_binds[i][2]] for i in range(N)]):
                break

        s_obj_positions, s_obj_rotations, s_obj_colors, s_obj_shapes, s_obj_sizes, s_obj_materials, \
                t_obj_positions, t_obj_rotations, t_obj_colors, t_obj_shapes, t_obj_sizes, t_obj_materials, \
                = apply_rule(object_binds, x, N, args.rule)

        success = render_scene(N,
                     s_obj_positions, s_obj_rotations, s_obj_colors, s_obj_shapes, s_obj_sizes, s_obj_materials,
                     t_obj_positions, t_obj_rotations, t_obj_colors, t_obj_shapes, t_obj_sizes, t_obj_materials,
                     args=args,
                     output_dir=sample_path,
                     target=not args.no_target)

        if not success:
            shutil.rmtree(sample_path, ignore_errors=True)
        else:
            shutil.move(sample_path, os.path.join(save_path, sample_dir))
            done += 1

        if done >= args.num_samples:
            break

    print('Dataset saved at : {}'.format(save_path))
