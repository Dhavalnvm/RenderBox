import bpy
import os
import math
import random
import uuid
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Vector

# ---------- CONFIG ----------
BLEND_DIR = r"C:/Users/Admin/Machine Learning/Propeller/blender"
OUTPUT_DIR = r"C:/Users/Admin/Machine Learning/Propeller/Output"
CLASS_NAME = 'propeller'
IMG_SIZE = 512
BUBBLE_COUNT = 60  # Reduced bubble count to avoid distraction

# Video settings - explicitly defined
FPS = 30
VIDEO_DURATION_SECONDS = 10
TOTAL_FRAMES = FPS * VIDEO_DURATION_SECONDS  # 300 frames for 10 seconds at 30fps
RPM = 1000  # 1000 RPM as requested

# ---------- UTILS ----------
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

# ---------- SCENE SETUP ----------
def setup_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam

def setup_lighting():
    # Main light to illuminate the propeller from behind
    key_light_data = bpy.data.lights.new(name="KeyLight", type='POINT')
    key_light = bpy.data.objects.new(name="KeyLight", object_data=key_light_data)
    bpy.context.collection.objects.link(key_light)
    
    # Position light behind and slightly above the propeller
    key_light.location = (0, 0.5, 0.5)
    key_light.data.energy = random.uniform(15, 20)  # Increased energy for better reflection on metal
    key_light.data.color = (0.9, 1.0, 0.7)  # Yellowish-green light
    
    # Rim light for edge definition
    rim_light_data = bpy.data.lights.new(name="RimLight", type='POINT')
    rim_light = bpy.data.objects.new(name="RimLight", object_data=rim_light_data)
    bpy.context.collection.objects.link(rim_light)
    rim_light.location = (-1.5, -1.5, 0.5)
    rim_light.data.energy = random.uniform(5, 7)  # Increased for better metallic highlights
    rim_light.data.color = (0.7, 0.9, 1.0)  # Blueish rim light
    
    # Add fill light to ensure propeller is clearly visible
    fill_light_data = bpy.data.lights.new(name="FillLight", type='POINT')
    fill_light = bpy.data.objects.new(name="FillLight", object_data=fill_light_data)
    bpy.context.collection.objects.link(fill_light)
    fill_light.location = (1.0, -1.0, 0.8)
    fill_light.data.energy = random.uniform(3, 4)
    fill_light.data.color = (0.8, 0.8, 1.0)  # Soft blue fill light

    return key_light

def setup_clear_underwater_world():
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for node in nodes:
        nodes.remove(node)

    # Create a darker blue background to match the image
    bg = nodes.new(type='ShaderNodeBackground')
    bg.inputs[0].default_value = (
        0.07,  # Very dark blue
        0.12,
        0.18,
        1.0)
    
    out = nodes.new(type='ShaderNodeOutputWorld')
    links.new(bg.outputs['Background'], out.inputs['Surface'])

# ---------- BUBBLES ----------
def add_bubbles(count=60):
    # Create a new collection for bubbles if it doesn't exist
    if "Bubbles" not in bpy.data.collections:
        bubbles_collection = bpy.data.collections.new("Bubbles")
        bpy.context.scene.collection.children.link(bubbles_collection)
    else:
        bubbles_collection = bpy.data.collections["Bubbles"]
    
    for _ in range(count):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=random.uniform(0.003, 0.01))
        b = bpy.context.object
        
        # Move bubble to bubbles collection
        for coll in b.users_collection:
            coll.objects.unlink(b)
        bubbles_collection.objects.link(b)
        
        # Set location - keeping bubbles more scattered but not in front of propeller
        # Place bubbles more to the sides and background, not blocking the view
        b.location = (
            random.uniform(-1.5, 1.5),
            random.uniform(-1.5, 1.5),
            random.uniform(-1, 2),
        )
        
        # Create material for bubble
        mat = bpy.data.materials.new(name=f"BubbleMat_{uuid.uuid4().hex[:4]}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        
        # Get the principled BSDF node
        principled = nodes.get("Principled BSDF")
        if principled:
            # Make bubble transparent - safely set properties
            principled.inputs["Base Color"].default_value = (0.8, 0.9, 1.0, 1.0)
            
            # Safely set properties (handle different Blender versions)
            try:
                principled.inputs["Metallic"].default_value = 0.0
                principled.inputs["Roughness"].default_value = 0.1
                principled.inputs["Transmission"].default_value = 0.95
                principled.inputs["IOR"].default_value = 1.33
            except: 
                pass
        
        b.data.materials.append(mat)
        
        # Animate bubbles rising
        b.keyframe_insert(data_path="location", frame=1)
        
        # Slower rising bubbles to match the mood of the image
        rise_speed = random.uniform(0.3, 0.8)
        b.location.z += rise_speed * VIDEO_DURATION_SECONDS
        
        # Add minimal horizontal drift
        b.location.x += random.uniform(-0.2, 0.2)
        b.location.y += random.uniform(-0.2, 0.2)
        
        b.keyframe_insert(data_path="location", frame=TOTAL_FRAMES)

# Function to create metallic material for propeller
def create_metallic_material():
    # Create new material
    mat_name = f"MetallicPropeller_{uuid.uuid4().hex[:4]}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create nodes for metallic shader
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    output = nodes.new(type='ShaderNodeOutputMaterial')
    
    # Connect nodes
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Set metallic properties
    principled.inputs['Base Color'].default_value = (0.8, 0.8, 0.85, 1.0)  # Slight blue tint for underwater look
    principled.inputs['Metallic'].default_value = 1.0  # Fully metallic
    principled.inputs['Roughness'].default_value = 0.1  # Very smooth/polished (low roughness)
    principled.inputs['Specular'].default_value = 0.5  # Medium specular intensity
    principled.inputs['Clearcoat'].default_value = 0.2  # Slight clearcoat for extra shine
    
    return mat

# ---------- MAIN ----------
def configure_video_settings():
    """Set up the render settings specifically for video output"""
    # Basic render settings
    bpy.context.scene.render.resolution_x = IMG_SIZE
    bpy.context.scene.render.resolution_y = IMG_SIZE
    bpy.context.scene.render.fps = FPS
    
    # Set up for VIDEO output
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    
    # FFMPEG settings
    ffmpeg = bpy.context.scene.render.ffmpeg
    ffmpeg.format = 'MPEG4'
    ffmpeg.codec = 'H264'
    ffmpeg.constant_rate_factor = 'MEDIUM'
    ffmpeg.gopsize = 18
    ffmpeg.video_bitrate = 8000
    
    # Frame range settings
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = TOTAL_FRAMES
    
    # Higher quality render settings if using EEVEE
    if bpy.context.scene.render.engine == 'BLENDER_EEVEE':
        bpy.context.scene.eevee.taa_render_samples = 128  # More samples for clearer metal reflections
        bpy.context.scene.eevee.use_gtao = True  # Ambient occlusion for better depth
        bpy.context.scene.eevee.use_bloom = True  # Bloom for light glow effect
        bpy.context.scene.eevee.bloom_intensity = 0.3  # Bloom strength
        bpy.context.scene.eevee.bloom_threshold = 0.8  # Bloom threshold
        
        # Enable screen space reflections for better metal appearance
        try:
            bpy.context.scene.eevee.use_ssr = True
            bpy.context.scene.eevee.use_ssr_refraction = True
            bpy.context.scene.eevee.ssr_quality = 1.0
            bpy.context.scene.eevee.ssr_max_roughness = 0.5
        except:
            pass
        
        try:
            # Try to activate volumetric effects if available
            bpy.context.scene.eevee.use_volumetric_lights = True
            bpy.context.scene.eevee.volumetric_end = 100
        except:
            pass
    
    # Higher quality render settings if using Cycles
    elif bpy.context.scene.render.engine == 'CYCLES':
        bpy.context.scene.cycles.samples = 256  # More samples for better metal quality
        bpy.context.scene.cycles.use_denoising = True
        try:
            bpy.context.scene.cycles.caustics_reflective = True
            bpy.context.scene.cycles.caustics_refractive = True
        except:
            pass

def main():
    # Ensure compatible render engine
    try:
        if bpy.context.scene.render.engine not in {'BLENDER_EEVEE', 'CYCLES'}:
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    except:
        try:
            bpy.context.scene.render.engine = 'CYCLES'
        except:
            print("Warning: Could not set a compatible render engine")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    blend_files = [f for f in os.listdir(BLEND_DIR) if f.endswith('.blend')]
    
    for blend_file in blend_files:
        try:
            # Try to open the file
            bpy.ops.wm.open_mainfile(filepath=os.path.join(BLEND_DIR, blend_file))
            
            # Set up render settings - MUST be done after loading each file
            configure_video_settings()
            
            # Setup scene elements
            key_light = setup_lighting()
            cam = setup_camera()
            setup_clear_underwater_world()
            add_bubbles(BUBBLE_COUNT)
            
            # Create metallic material
            metallic_material = create_metallic_material()

            propeller_found = False
            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH' and "Bubbles" not in obj.name:
                    # Skip collection objects by checking name
                    
                    # Center object, normalize size
                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                    
                    try:
                        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                        
                        # Avoid division by zero
                        max_dim = max(obj.dimensions) if max(obj.dimensions) > 0 else 1.0
                        scale = 1.0 / max_dim
                        obj.scale = (scale, scale, scale)
                        bpy.ops.object.transform_apply(scale=True)
                    except Exception as e:
                        print(f"Warning: Could not transform object: {str(e)}")
                    
                    # Apply metallic material to propeller
                    # First remove all materials
                    obj.data.materials.clear()
                    # Then assign our metallic material
                    obj.data.materials.append(metallic_material)

                    # Calculate rotation for 1000 RPM over 10 seconds
                    total_rotation = (RPM / 60) * VIDEO_DURATION_SECONDS * 360  # in degrees
                    
                    # Animate propeller
                    obj.animation_data_clear()
                    try: 
                        obj.driver_remove("rotation_euler", 0)
                    except: 
                        pass
                    
                    obj.rotation_euler = (0.0, 0.0, 0.0)
                    obj.keyframe_insert(data_path="rotation_euler", frame=1)
                    
                    obj.rotation_euler = (math.radians(total_rotation), 0.0, 0.0)
                    obj.keyframe_insert(data_path="rotation_euler", frame=TOTAL_FRAMES)
                    
                    # Set linear interpolation for smooth rotation
                    if obj.animation_data and obj.animation_data.action:
                        for fc in obj.animation_data.action.fcurves:
                            for kp in fc.keyframe_points:
                                kp.interpolation = 'LINEAR'

                    # Position camera to match the reference image angle
                    cam.location = (2.2, -2.0, 0.0)  # Position to see propeller similar to image
                    cam.data.lens = random.uniform(28, 32)  # Similar lens settings
                    
                    # Calculate the direction vector from camera to object
                    target_position = Vector((0, 0, 0))  # Object at origin
                    look_direction = target_position - cam.location
                    
                    # Set camera to look at object
                    cam.rotation_euler = look_direction.to_track_quat('-Z', 'Y').to_euler()
                    
                    # Set DOF settings for better depth and focus on object
                    try:
                        cam.data.dof.use_dof = True
                        cam.data.dof.focus_distance = cam.location.length  # Focus on object
                        cam.data.dof.aperture_fstop = random.uniform(2.8, 4.0)  # Lower f-stop for better subject isolation
                    except:
                        print("Warning: Could not set DOF settings")

                    # Set video output path - use blend file name as part of the output
                    blend_name = os.path.splitext(blend_file)[0]
                    video_path = os.path.join(OUTPUT_DIR, f"{blend_name}_1000rpm_metallic_{CLASS_NAME}_{uuid.uuid4().hex[:4]}.mp4")
                    
                    # CRITICAL: Make sure the output path explicitly includes the file extension
                    bpy.context.scene.render.filepath = video_path
                    
                    print(f"Rendering video: {video_path}")
                    print(f"Propeller RPM: {RPM}")
                    print(f"Camera: Dark underwater with metallic propeller")
                    print(f"Frame range: {bpy.context.scene.frame_start} to {bpy.context.scene.frame_end}")
                    print(f"File format: {bpy.context.scene.render.image_settings.file_format}")
                    
                    # Render animation as video
                    bpy.ops.render.render(animation=True)
                    propeller_found = True
                    break  # Process only the first suitable mesh object
            
            if not propeller_found:
                print(f"Warning: No suitable mesh object found in {blend_file}")
                
        except Exception as e:
            print(f"Error processing {blend_file}: {str(e)}")
            continue

main()
