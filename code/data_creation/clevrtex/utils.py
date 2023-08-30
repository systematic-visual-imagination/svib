# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import os
import sys

from pathlib import Path

import bpy
import bpy_extras

"""
Some utility functions for interacting with Blender
"""


def extract_args(input_argv=None):
    """
  Pull out command-line arguments after "--". Blender ignores command-line flags
  after --, so this lets us forward command line arguments from the blender
  invocation to our own script.
  """
    if input_argv is None:
        input_argv = sys.argv
    output_argv = []
    if '--' in input_argv:
        idx = input_argv.index('--')
        output_argv = input_argv[(idx + 1):]
    return output_argv


def parse_args(parser, argv=None):
    return parser.parse_args(extract_args(argv))


# I wonder if there's a better way to do this?
def delete_object(obj):
    """ Delete a specified blender object """
    for o in bpy.data.objects:
        o.select_set(state=False, view_layer=bpy.context.view_layer)
    obj.select_set(state=True, view_layer=bpy.context.view_layer)
    bpy.ops.object.delete()


def get_camera_coords(cam, pos):
    """
  For a specified point, get both the 3D coordinates and 2D pixel-space
  coordinates of the point from the perspective of the camera.

  Inputs:
  - cam: Camera object
  - pos: Vector giving 3D world-space position

  Returns a tuple of:
  - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
    in the range [-1, 1]
  """
    scene = bpy.context.scene
    x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
    scale = scene.render.resolution_percentage / 100.0
    w = int(scale * scene.render.resolution_x)
    h = int(scale * scene.render.resolution_y)
    px = int(round(x * w))
    py = int(round(h - y * h))
    return px, py, z

def set_layer(obj, layer_idx):
  """ Move an object to a particular layer """
  # Set the target layer to True first because an object must always be on
  # at least one layer.
  obj.layers[layer_idx] = True
  for i in range(len(obj.layers)):
    obj.layers[i] = (i == layer_idx)


def add_object(object_dir, name, scale, loc, theta=0):
    """
  Load an object from a file. We assume that in the directory object_dir, there
  is a file named "$name.blend" which contains a single object named "$name"
  that has unit size and is centered at the origin.

  - scale: scalar giving the size that the object should be in the scene
  - loc: tuple (x, y) giving the coordinates on the ground plane where the
    object should be placed.
  """
    # First figure out how many of this object are already in the scene so we can
    # give the new object a unique name
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
    bpy.ops.wm.append(filename=filename)

    # Give it a new name to avoid conflicts
    new_name = '%s_%d' % (name, count)
    bpy.data.objects[name].name = new_name

    # Set the new object as active, then rotate, scale, and translate it
    x, y = loc
    o = bpy.data.objects[new_name]
    o.select_set(state=True, view_layer=bpy.context.view_layer)
    bpy.context.view_layer.objects.active = o
    bpy.context.object.rotation_euler[2] = theta
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.transform.translate(value=(x, y, scale))
    o.select_set(state=False, view_layer=bpy.context.view_layer)


def load_materials(material_dir):
    """
  Load materials from a directory. We assume that the directory contains .blend
  files with one material each. The file X.blend has a single NodeTree item named
  X; this NodeTree item must have a "Color" input that accepts an RGBA value.
  """
    for fn in os.listdir(material_dir):
        if not fn.endswith('.blend'): continue
        name = os.path.splitext(fn)[0]
        filepath = os.path.join(material_dir, fn, 'NodeTree', name)
        bpy.ops.wm.append(filename=filepath)


def load_material(path):
    path = Path(path)
    filepath = path / 'NodeTree' / path.stem
    bpy.ops.wm.append(filename=str(filepath))


def add_material(obj, mat_path, **properties):
    """
    Create a new material and assign it to the active object. "name" should be the
    name of a material that has been previously loaded using load_materials.
    """
    mat_path = Path(mat_path)
    print('mat_path = ', mat_path)
    # Sometime Displacement is called Displacement Strength
    if 'Displacement' in properties:
        properties['Displacement Strength'] = properties['Displacement']

    # Figure out how many materials are already in the scene
    mat_count = len(bpy.data.materials)
    names = {m.name for m in bpy.data.materials}
    name = mat_path.stem
    mat_name = mat_path.stem
    if name in names:
        idx = sum(1 for m in bpy.data.materials if m.name.startswith(name))
        mat_name = name + f'_{idx + 1}'

    # Create a new material
    mat = bpy.data.materials.new(mat_name)
    mat.name = mat_name
    mat.use_nodes = True
    mat.cycles.displacement_method = 'BOTH'

    # Attach the new material to the object
    # Make sure it doesn't already have materials
    assert len(
        obj.data.materials) == 0, f"{obj.name} has multiple materials ({', '.join(m.name for m in obj.data.materials if m is not None)}), adding {name} will fail"
    obj.data.materials.append(mat)

    mat.node_tree.links.clear()
    mat.node_tree.nodes.clear()

    output_node = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    output_node.is_active_output = True

    # Add a new GroupNode to the node tree of the active material,

    group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
    if name not in bpy.data.node_groups:
        load_material(mat_path)
    group_node.node_tree = bpy.data.node_groups[name]
    # Also this seems to be the only way to copy a node tree in the headless mode

    # Wire first by-name then by preset names, to the group outputs to the material output
    for out_socket in group_node.outputs:
        if out_socket.name in output_node.inputs:
            mat.node_tree.links.new(
                group_node.outputs[out_socket.name],
                output_node.inputs[out_socket.name],
            )
        else:
            # print(f"{out_socket.name} not found in the output of the material")
            pass
        if not output_node.inputs['Surface'].is_linked:
            if 'Shader' in group_node.outputs and not group_node.outputs['Shader'].is_linked:
                # print(f"Unlinked Surface socket in the material output; trying to fill with Shader socket of the group")
                mat.node_tree.links.new(
                    group_node.outputs["Shader"],
                    output_node.inputs["Surface"],
                )
            elif 'BSDF' in group_node.outputs and not group_node.outputs['BSDF'].is_linked:
                # print(f"Unlinked Surface socket in the material output; trying to fill with BSDF socket of the group")
                mat.node_tree.links.new(
                    group_node.outputs["BSDF"],
                    output_node.inputs["Surface"],
                )
            else:
                raise ValueError(f"Cannot resolve material output for {mat.name}")

    for inp in group_node.inputs:
        if inp.name in properties:
            inp.default_value = properties[inp.name]

    return mat


def add_shadeless_nodes_to_material(mat, shadeless_clr):
    """
    Inject nodes to the material tree required for rendering solid colour output
    """
    mix_node = mat.node_tree.nodes.new('ShaderNodeMixShader')
    mix_node.name = 'InjectedShadelessMix'
    dif_node = mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    dif_node.name = 'InjectedShadelessDif'
    lit_node = mat.node_tree.nodes.new('ShaderNodeLightPath')
    lit_node.name = 'InjectedShadelessLit'
    emi_node = mat.node_tree.nodes.new('ShaderNodeEmission')
    emi_node.name = 'InjectedShadelessEmission'
    emi_node.inputs['Color'].default_value = shadeless_clr
    l1 = mat.node_tree.links.new(
        lit_node.outputs['Is Camera Ray'],
        mix_node.inputs['Fac'],
    )
    l2 = mat.node_tree.links.new(
        dif_node.outputs['BSDF'],
        mix_node.inputs[1],
    )
    l3 = mat.node_tree.links.new(
        emi_node.outputs['Emission'],
        mix_node.inputs[2],
    )
    return mix_node, (mix_node, dif_node, lit_node, emi_node), (l1, l2, l3)


def set_to_shadeless(mat, shadeless_clr):
    """
    Rewire the material to output solid colour <shadeless_clr>.
    Returns a callback that returns the material to original state
    """
    # print(f"Setting {mat.name} to shadeless")
    # Locate output node
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'Material Output':
            output_node = n
            break
    else:
        raise ValueError(f"Could not locate output node in {mat.name} Material")

    socket = None
    if output_node.inputs['Surface'].is_linked:
        l = None
        for link in mat.node_tree.links:
            if link.to_socket == output_node.inputs['Surface']:
                l = link
                break
        else:
            raise ValueError(f"Could not locate output node Surface link in {mat.name} Material")
        socket = l.from_socket.node.outputs[l.from_socket.name]  # Lets hope there not multiple with the same name
        # print(f"Will try to restore link between {l.from_socket.node.name}:{l.from_socket.name} {socket}")
        mat.node_tree.links.remove(l)

    # Check that shadeless rendering nodes have not already been injected to this material
    mix_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'InjectedShadelessMix':
            mix_node = n
            break
    if mix_node is None:
        # Inject the nodes
        mix_node, nodes, links = add_shadeless_nodes_to_material(mat, shadeless_clr)
    else:
        # If they already exist; just set the colour to the correct value
        nodes = []
        for n in mat.node_tree.nodes:
            if n.name == 'InjectedShadelessEmission':
                n.inputs['Color'].default_value = shadeless_clr
            if n.name.startswith('InjectedShadeless'):
                nodes.append(n)
        links = set()
        for n in nodes:
            for s in n.inputs:
                if s.is_linked:
                    for l in s.links:
                        links.add(l)

    # Check and correct the node connection
    if mix_node.outputs['Shader'].is_linked:
        # Check and reset the node links between Shadeless mix node and the material output
        offending_link = None
        for link in mat.node_tree.links:
            if link.from_socker.node == mix_node:
                offending_link = link
                break
        else:
            raise ValueError(f"Could not locate offending mix_shader link in the {mat.name} Material")
        mat.node_tree.links.remove(offending_link)

    temp_link = mat.node_tree.links.new(
        mix_node.outputs['Shader'],
        output_node.inputs['Surface'],
    )

    return (
        mat,
        temp_link,
        socket,
        output_node,
        links,
        nodes
    )

def undo_shadeless(mat, temp_link, socket, output_node, links, nodes):
    mat.node_tree.links.remove(temp_link)
    if socket:
        mat.node_tree.links.new(socket, output_node.inputs['Surface'])
    # print(f"Reverting {mat.name} to the original")
    for l in links:
        mat.node_tree.links.remove(l)
    for n in nodes:
        mat.node_tree.nodes.remove(n)
