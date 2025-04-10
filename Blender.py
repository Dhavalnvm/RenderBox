import bpy
import os
import math
import random
import uuid
import cv2
import numpy as np
from bpy_extras.object_utils import world_to_camera_view

# ---------- CONFIG ----------
BLEND_DIR = r"C:/Users/ADMIN/PycharmProjects/dataset generator/blender"
OUTPUT_DIR = r"C:/Users/ADMIN/PycharmProjects/dataset generator/output dir/data3"
CLASS_NAME = 'propeller'
IMG_SIZE = 512
NUM_IMAGES_PER_FILE = 40
TRAIN_RATIO = 0.8

# ---------- SETUP ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_TRAIN = os.path.join(OUTPUT_DIR, 'images/train')
IMG_VAL = os.path.join(OUTPUT_DIR, 'images/val')
LBL_TRAIN = os.path.join(OUTPUT_DIR, 'labels/train')
LBL_VAL = os.path.join(OUTPUT_DIR, 'labels/val')
for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    os.makedirs(d, exist_ok=True)

# ---------- CAMERA & LIGHT SETUP ----------
def setup_camera():
    if 'Camera' in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects['Camera'], do_unlink=True)
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam

def setup_lighting():
    for light in [obj for obj in bpy.data.objects if obj.type == 'LIGHT']:
        bpy.data.objects.remove(light, do_unlink=True)
    light_data = bpy.data.lights.new(name="light", type='SUN')
    light_obj = bpy.data.objects.new(name="light", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (5, -5, 5)

def setup_underwater_environment():
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    for node in nodes:
        nodes.remove(node)

    bg_node = nodes.new(type='ShaderNodeBackground')
    bg_node.inputs[0].default_value = (0.0, 0.1, 0.2, 1.0)  # dark bluish
    bg_node.inputs[1].default_value = 1.0

    volume_node = nodes.new(type='ShaderNodeVolumeScatter')
    volume_node.inputs[0].default_value = (0.2, 0.3, 0.6, 1.0)
    volume_node.inputs[1].default_value = 0.1

    output_node = nodes.new(type='ShaderNodeOutputWorld')
    links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    links.new(volume_node.outputs['Volume'], output_node.inputs['Volume'])

# ---------- RENDER & LABEL ----------
def render_and_save(obj, out_prefix, angle_idx, is_train):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    dims = obj.dimensions
    max_dim = max(dims)
    if max_dim > 0:
        scale_factor = 1.0 / max_dim
        obj.scale = (scale_factor, scale_factor, scale_factor)
        bpy.ops.object.transform_apply(scale=True)

    obj.rotation_euler = (
        math.radians(random.uniform(0, 360)),
        math.radians(random.uniform(0, 360)),
        math.radians(random.uniform(0, 360))
    )

    radius = 3.5
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.radians(45), math.radians(135))
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)

    cam = bpy.context.scene.camera
    cam.location = (x, y, z)
    direction = obj.location - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    img_name = f"{out_prefix}_{angle_idx}.png"
    img_path = os.path.join(IMG_TRAIN if is_train else IMG_VAL, img_name)
    bpy.context.scene.render.filepath = img_path
    bpy.ops.render.render(write_still=True)

    mat_world = obj.matrix_world
    coords_2d = [world_to_camera_view(bpy.context.scene, cam, mat_world @ v.co)
                 for v in obj.data.vertices]
    coords_2d = [(int(p.x * IMG_SIZE), int(IMG_SIZE - p.y * IMG_SIZE)) for p in coords_2d if
                 0 <= p.x <= 1 and 0 <= p.y <= 1]

    if not coords_2d:
        return

    x_coords = [p[0] for p in coords_2d]
    y_coords = [p[1] for p in coords_2d]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    x_center = ((x_min + x_max) / 2) / IMG_SIZE
    y_center = ((y_min + y_max) / 2) / IMG_SIZE
    box_w = (x_max - x_min) / IMG_SIZE
    box_h = (y_max - y_min) / IMG_SIZE

    label_dir = LBL_TRAIN if is_train else LBL_VAL
    label_path = os.path.join(label_dir, img_name.replace('.png', '.txt'))
    with open(label_path, 'w') as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

# ---------- MAIN PROCESS ----------
def main():
    bpy.context.scene.render.resolution_x = IMG_SIZE
    bpy.context.scene.render.resolution_y = IMG_SIZE
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    setup_lighting()
    cam = setup_camera()
    setup_underwater_environment()

    blend_files = [f for f in os.listdir(BLEND_DIR) if f.endswith('.blend')]
    random.shuffle(blend_files)
    total_files = len(blend_files)
    train_cutoff = int(TRAIN_RATIO * total_files)

    for i, blend_file in enumerate(blend_files):
        is_train = i < train_cutoff
        blend_path = os.path.join(BLEND_DIR, blend_file)
        bpy.ops.wm.open_mainfile(filepath=blend_path)
        setup_underwater_environment()
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                uid = uuid.uuid4().hex[:8]
                for j in range(NUM_IMAGES_PER_FILE):
                    render_and_save(obj, f"{uid}_{CLASS_NAME}", j, is_train)

main()
