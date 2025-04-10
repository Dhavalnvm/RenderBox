import os
import cv2
import uuid
import trimesh
import pyrender
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import albumentations as A
from glob import glob
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# ---------- CONFIG ----------
STL_DIR = r"C:\Users\ADMIN\PycharmProjects\dataset generator\STL models"
BASE_OUTPUT = r"C:\Users\ADMIN\PycharmProjects\dataset generator\output dir"
SINGLE_CLASS = True
CLASS_NAME = 'propeller'
IMG_SIZE = 512
NUM_IMAGES_PER_FILE = 40
TRAIN_RATIO = 0.8
# Logging and diagnostics
SAVE_DIAGNOSTICS = True
LOG_LEVEL = logging.INFO

# ---------- SETUP LOGGING ----------
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_dir = os.path.join(BASE_OUTPUT, f'run_{timestamp}')
log_dir = os.path.join(base_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'generator.log'))
    ]
)
logger = logging.getLogger('STLRenderer')

# ---------- OUTPUT SETUP ----------
img_train_dir = os.path.join(base_dir, 'images/train')
img_val_dir = os.path.join(base_dir, 'images/val')
lbl_train_dir = os.path.join(base_dir, 'labels/train')
lbl_val_dir = os.path.join(base_dir, 'labels/val')
if SAVE_DIAGNOSTICS:
    DIAGNOSTICS_DIR = os.path.join(base_dir, 'diagnostics')
    os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

for directory in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
    os.makedirs(directory, exist_ok=True)

logger.info(f"Output directory: {base_dir}")
logger.info(f"Rendering {NUM_IMAGES_PER_FILE} images per STL file")

# ---------- AUGMENTATION ----------
augment = A.Compose([
    A.GaussianBlur(3, p=0.5),
    A.RandomBrightnessContrast(0.2, 0.3, p=0.7),
    A.HueSaturationValue(20, 50, 10, p=0.8),
    A.RGBShift(20, 30, 60, p=0.6),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.OpticalDistortion(0.2, 0.2, p=0.4)
])


def apply_effects(rgb_img):
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    return augment(image=bgr)["image"]


# ---------- MESH PREPARATION ----------
def prepare_mesh(mesh_path):
    """Load and prepare a mesh for rendering"""
    try:
        # Try loading with trimesh
        loaded_mesh = trimesh.load(mesh_path)

        if not isinstance(loaded_mesh, trimesh.Trimesh):
            # Handle scene type (multiple meshes)
            if isinstance(loaded_mesh, trimesh.Scene):
                # Extract the geometry from the scene
                geom = list(loaded_mesh.geometry.values())
                if len(geom) == 0:
                    logger.error(f"No geometry found in scene at {mesh_path}")
                    return None

                # Combine all meshes into one
                vertices = []
                faces = []
                face_offset = 0

                for g in geom:
                    if isinstance(g, trimesh.Trimesh):
                        vertices.append(g.vertices)
                        faces.append(g.faces + face_offset)
                        face_offset += len(g.vertices)

                if not vertices:
                    logger.error(f"No valid meshes found in scene at {mesh_path}")
                    return None

                loaded_mesh = trimesh.Trimesh(
                    vertices=np.vstack(vertices),
                    faces=np.vstack(faces)
                )
            else:
                logger.error(f"Unsupported mesh type: {type(loaded_mesh)}")
                return None

        # Center the mesh
        loaded_mesh.apply_translation(-loaded_mesh.centroid)

        # Check mesh integrity
        if not loaded_mesh.is_watertight:
            logger.warning(f"Mesh at {mesh_path} is not watertight, fixing normals...")
            loaded_mesh.fix_normals()

        # Scale the mesh to fit within a unit cube
        extents = loaded_mesh.extents
        if np.any(extents <= 0) or np.all(np.isclose(extents, 0)):
            logger.error(f"Invalid mesh extents: {extents}")
            return None

        scale_factor = 0.8 / max(extents)  # Leave some margin around the object
        loaded_mesh.apply_scale(scale_factor)

        return loaded_mesh

    except Exception as e:
        logger.error(f"Failed to load mesh {mesh_path}: {e}")
        return None


# ---------- CAMERA UTILS ----------
def look_at_pose(cam_pos, target=np.array([0, 0, 0])):
    """Generate a camera pose looking at a target"""
    forward = target - cam_pos
    forward_norm = np.linalg.norm(forward)

    if forward_norm < 1e-10:
        # Handle the case where camera is at the target
        forward = np.array([0, 0, -1])
    else:
        forward = forward / forward_norm

    world_up = np.array([0, 1, 0])

    # Handle case where camera is aligned with up vector
    if abs(np.dot(forward, world_up)) > 0.999:
        # Choose a different up vector
        world_up = np.array([1, 0, 0])

    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)

    up = np.cross(forward, right)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = forward
    pose[:3, 3] = cam_pos

    return pose


# ---------- DIAGNOSTIC VISUALIZATION ----------
def save_diagnostic_images(img_rgb, depth_map, mask, segmentation_mask, detections, img_name):
    """Save diagnostic images showing the rendering and detection results"""
    if not SAVE_DIAGNOSTICS:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # RGB image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("RGB Render")
    axes[0, 0].axis('off')

    # Depth map
    if depth_map is not None:
        norm_depth = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        axes[0, 1].imshow(norm_depth, cmap='viridis')
        axes[0, 1].set_title("Depth Map")
    else:
        axes[0, 1].imshow(np.zeros((IMG_SIZE, IMG_SIZE)), cmap='viridis')
        axes[0, 1].set_title("No Depth Data")
    axes[0, 1].axis('off')

    # Processed mask
    if mask is not None:
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title("Processed Mask")
    else:
        axes[0, 2].imshow(np.zeros((IMG_SIZE, IMG_SIZE)), cmap='gray')
        axes[0, 2].set_title("No Mask Data")
    axes[0, 2].axis('off')

    # Segmentation mask
    if segmentation_mask is not None:
        axes[1, 0].imshow(segmentation_mask, cmap='gray')
        axes[1, 0].set_title("Segmentation Mask")
    else:
        axes[1, 0].imshow(np.zeros((IMG_SIZE, IMG_SIZE)), cmap='gray')
        axes[1, 0].set_title("No Segmentation Data")
    axes[1, 0].axis('off')

    # RGB with all bounding boxes
    preview_img = img_rgb.copy()
    axes[1, 1].imshow(preview_img)

    if detections and len(detections) > 0:
        import matplotlib.patches as patches
        for i, bbox in enumerate(detections):
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2,
                    edgecolor=['r', 'g', 'b', 'y'][i % 4],
                    facecolor='none'
                )
                axes[1, 1].add_patch(rect)
        axes[1, 1].set_title(f"{len([b for b in detections if b is not None])} Detections")
    else:
        axes[1, 1].set_title("No Detections")
    axes[1, 1].axis('off')

    # Empty or extra info
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, f"{img_name}_diagnostic.png"))
    plt.close(fig)


# ---------- IMPROVED DETECTION METHODS ----------
def get_bbox_from_depth(depth_map):
    """Get bounding box from depth map"""
    if depth_map is None or np.all(depth_map == 0) or np.max(depth_map) < 1e-6:
        return None, None

    # Create binary mask
    # Use dynamic thresholding based on depth values present
    depth_values = depth_map[depth_map > 0]
    if len(depth_values) == 0:
        return None, None

    # Try to distinguish foreground from background
    object_mask = (depth_map > 0).astype(np.uint8) * 255

    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, object_mask

    # Filter out noise by excluding very small contours
    valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
    if not valid_contours:
        return None, object_mask

    # Get the largest contour
    largest_contour = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Return the bounding box and mask
    return (x, y, x + w, y + h), object_mask


def get_bbox_from_rgb(img_rgb):
    """Get bounding box from RGB image"""
    if img_rgb is None:
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )

    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask

    # Filter out noise by excluding very small contours
    valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
    if not valid_contours:
        return None, mask

    # Get the largest contour
    largest_contour = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Return the bounding box and mask
    return (x, y, x + w, y + h), mask


def get_bbox_from_segmentation(scene, mesh_node, render_flags=None):
    """Get bounding box using pyrender's segmentation feature"""
    if scene is None or mesh_node is None:
        return None, None

    try:
        # Create a renderer with segmentation capability
        r = pyrender.OffscreenRenderer(IMG_SIZE, IMG_SIZE)

        # Set flags for segmentation rendering
        flags = pyrender.RenderFlags.RGBA
        if render_flags:
            flags |= render_flags

        # Render segmentation
        seg_img = r.render(scene, flags=flags)
        r.delete()

        # Get the unique mesh colors
        if seg_img.shape[2] == 4:  # RGBA
            colors = seg_img[:, :, :3]
        else:
            colors = seg_img

        # Create a binary mask where the object is
        # The object will have a unique color different from background
        unique_colors = np.unique(colors.reshape(-1, colors.shape[2]), axis=0)

        if len(unique_colors) <= 1:
            # No object visible
            return None, None

        # Create mask for each unique color
        masks = []
        for color in unique_colors:
            if not np.all(color == 0):  # Skip black background
                mask = np.all(colors == color.reshape(1, 1, 3), axis=2).astype(np.uint8) * 255
                masks.append(mask)

        if not masks:
            return None, None

        # Combine masks if multiple
        combined_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, combined_mask

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        return (x, y, x + w, y + h), combined_mask

    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        return None, None


def get_bbox_from_edges(img_rgb):
    """Get bounding box by detecting edges in the RGB image"""
    if img_rgb is None:
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Dilate edges to connect them
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, dilated

    # Filter out tiny contours
    valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
    if not valid_contours:
        return None, dilated

    # Get the largest contour
    largest_contour = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return (x, y, x + w, y + h), dilated


# ---------- YOLO SAVING ----------
def save_yolo(img, bbox, name_base, is_train):
    """Save image and annotation in YOLO format"""
    h, w, _ = img.shape
    x_center = ((bbox[0] + bbox[2]) / 2) / w
    y_center = ((bbox[1] + bbox[3]) / 2) / h
    bw = (bbox[2] - bbox[0]) / w
    bh = (bbox[3] - bbox[1]) / h
    label = f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n"

    img_folder = img_train_dir if is_train else img_val_dir
    lbl_folder = lbl_train_dir if is_train else lbl_val_dir
    img_path = os.path.join(img_folder, f"{name_base}.png")
    lbl_path = os.path.join(lbl_folder, f"{name_base}.txt")

    cv2.imwrite(img_path, img)
    with open(lbl_path, 'w') as f:
        f.write(label)


# ---------- YOLO DATASET.YAML ----------
def generate_yaml():
    """Generate dataset.yaml for YOLO training"""
    yaml_path = os.path.join(base_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {base_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"names: ['{CLASS_NAME}']\n")


# ---------- MAIN RENDERING FUNCTION ----------
def render_mesh(mesh_path, model_name, img_idx):
    """Render a single view of a mesh with multiple detection methods"""
    # Load and prepare mesh
    loaded_mesh = prepare_mesh(mesh_path)
    if loaded_mesh is None:
        return None

    # Create a randomized orientation
    rotation = trimesh.transformations.random_rotation_matrix()
    rotated_mesh = loaded_mesh.copy()
    rotated_mesh.apply_transform(rotation)

    # Convert to pyrender mesh
    try:
        mesh = pyrender.Mesh.from_trimesh(rotated_mesh, smooth=True)
    except Exception as e:
        logger.error(f"Failed to convert mesh to pyrender: {e}")
        return None

    # Create a scene with better lighting
    scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[0.2, 0.2, 0.2])
    mesh_node = scene.add(mesh)

    # Compute camera parameters
    bbox_diag = np.linalg.norm(rotated_mesh.extents)
    fov = np.pi / 3.0
    cam_distance = (bbox_diag / 2) / np.tan(fov / 2) + 0.5

    # Randomize camera position on a sphere around the object
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.random.uniform(0, np.pi)

    x = cam_distance * np.sin(theta) * np.cos(phi)
    y = cam_distance * np.sin(theta) * np.sin(phi)
    z = cam_distance * np.cos(theta)

    cam_pos = np.array([x, y, z])
    cam_pose = look_at_pose(cam_pos)

    # Add camera
    camera = pyrender.PerspectiveCamera(yfov=fov)
    scene.add(camera, pose=cam_pose)

    # Add lights from multiple directions
    # Main light from camera
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)

    # Add point lights at different positions
    light_positions = [
        [1, 1, 1],
        [-1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1]
    ]

    for lp in light_positions:
        lp = np.array(lp) * cam_distance * 0.7
        scene.add(pyrender.PointLight(color=np.ones(3), intensity=2.0),
                  pose=look_at_pose(lp))

    # Render
    try:
        renderer = pyrender.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
        img_rgb, depth_map = renderer.render(scene)
        renderer.delete()
    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        return None

    # Use multiple detection methods
    detections = []

    # Method 1: Depth map detection
    bbox_depth, mask_depth = get_bbox_from_depth(depth_map)
    detections.append(bbox_depth)

    # Method 2: RGB-based detection
    bbox_rgb, mask_rgb = get_bbox_from_rgb(img_rgb)
    detections.append(bbox_rgb)

    # Method 3: Edge detection
    bbox_edge, mask_edge = get_bbox_from_edges(img_rgb)
    detections.append(bbox_edge)

    # Method 4: Segmentation-based detection (if supported)
    bbox_seg, mask_seg = get_bbox_from_segmentation(scene, mesh_node)
    detections.append(bbox_seg)

    # Choose best bounding box (non-None and largest area)
    valid_bboxes = [bbox for bbox in detections if bbox is not None]

    if not valid_bboxes:
        logger.warning(f"No detection for {model_name} [{img_idx}] with any method")
        save_diagnostic_images(img_rgb, depth_map, mask_depth, mask_seg, detections, f"{model_name}_{img_idx}_nodetect")
        return None

    # Select bbox with largest area
    best_bbox = max(valid_bboxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    # Validate size ratio
    box_area = (best_bbox[2] - best_bbox[0]) * (best_bbox[3] - best_bbox[1])
    img_area = IMG_SIZE * IMG_SIZE
    ratio = box_area / img_area

    if ratio < 0.02:  # Very lenient minimum size
        logger.warning(f"Bounding box too small for {model_name} [{img_idx}]: {ratio:.4f}")
        save_diagnostic_images(img_rgb, depth_map, mask_depth, mask_seg, detections, f"{model_name}_{img_idx}_small")
        return None

    if ratio > 0.99:  # Almost entire image
        logger.warning(f"Bounding box too large for {model_name} [{img_idx}]: {ratio:.4f}")
        save_diagnostic_images(img_rgb, depth_map, mask_depth, mask_seg, detections, f"{model_name}_{img_idx}_large")
        return None

    # Apply visual effects
    img_bgr = apply_effects(img_rgb)

    # Create unique name
    name = f"{uuid.uuid4().hex[:8]}_{model_name}_{img_idx}"

    # Save diagnostic image if detection was successful
    save_diagnostic_images(img_rgb, depth_map, mask_depth, mask_seg, detections, f"{name}")

    return img_bgr, best_bbox, name


# ---------- MAIN PROCESS ----------
def generate_dataset():
    """Main function to generate the dataset"""
    stl_files = sorted(glob(os.path.join(STL_DIR, '*.stl')))
    if not stl_files:
        logger.error(f"No STL files found in {STL_DIR}")
        return

    logger.info(f"Found {len(stl_files)} STL files for class: {CLASS_NAME}")
    for f in stl_files:
        logger.info(f" - {Path(f).name}")

    dataset = []

    for stl_path in stl_files:
        mesh_name = Path(stl_path).stem
        logger.info(f"Processing {mesh_name}...")

        success_count = 0
        for i in range(NUM_IMAGES_PER_FILE):
            try:
                result = render_mesh(stl_path, mesh_name, i)
                if result is not None:
                    img_bgr, bbox, name = result
                    dataset.append((img_bgr, bbox, name))
                    success_count += 1

                    if success_count % 5 == 0:
                        logger.info(f"Generated {success_count}/{NUM_IMAGES_PER_FILE} images for {mesh_name}")
            except Exception as e:
                logger.error(f"Error processing {mesh_name} [{i}]: {e}")
                continue

        logger.info(f"Completed {mesh_name}: {success_count}/{NUM_IMAGES_PER_FILE} successful images")

    if not dataset:
        logger.error("No valid renderings were created. Check STL files and rendering settings.")
        return

    logger.info(f"Generated {len(dataset)} total valid images across all models")

    # Split into train/val sets
    train_data, val_data = train_test_split(dataset, train_size=TRAIN_RATIO, shuffle=True)

    logger.info(f"Saving {len(train_data)} training images and {len(val_data)} validation images")

    # Save images and labels
    for img, bbox, name in train_data:
        save_yolo(img, bbox, name, is_train=True)
    for img, bbox, name in val_data:
        save_yolo(img, bbox, name, is_train=False)

    # Generate YAML
    generate_yaml()

    logger.info(f"Dataset generation complete!")
    logger.info(f"Output directory: {base_dir}")
    if SAVE_DIAGNOSTICS:
        logger.info(f"Diagnostic images saved to: {DIAGNOSTICS_DIR}")


# ---------- RUN GENERATOR ----------
if __name__ == "__main__":
    generate_dataset()