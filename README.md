# RenderBox
# STL-Based Synthetic Dataset Generator for Object Detection (YOLO Format)

This project generates a **synthetic underwater dataset** from STL 3D models using **PyRender** and **Trimesh**, with automatic **bounding box detection** and **YOLO annotation**. It applies data augmentation and exports images, labels, and a YOLO-compatible `dataset.yaml`.

## ✅ Features

- Loads and processes `.stl` 3D models
- Renders randomized views using `pyrender`
- Applies 4 detection strategies:
  - Depth-based bounding box
  - RGB-based detection
  - Edge-based detection
  - Segmentation via renderer
- Applies underwater-like effects using `albumentations`
- Exports training/validation datasets in YOLO format
- Logs diagnostics and optionally saves visual debug images
- Automatically generates `dataset.yaml`

## 📁 Directory Structure

<pre> ```bash output/ └── run_YYYYMMDD_HHMMSS/ # Unique run folder timestamped at generation ├── images/ # Contains generated images │ ├── train/ # Training images │ └── val/ # Validation images ├── labels/ # Corresponding YOLO-format annotation files │ ├── train/ # Training labels │ └── val/ # Validation labels ├── logs/ # Additional logs or metadata ├── generator.log # Log file with generation summary ├── diagnostics/ # (optional) Visual/debugging outputs for inspection └── dataset.yaml # YOLO-compatible dataset configuration file ``` </pre>


##🔧 Configuration

STL_DIR = "path/to/STL models"         # Input STL folder
BASE_OUTPUT = "path/to/output dir"     # Output folder
CLASS_NAME = 'propeller'               # Class name (single class setup)
IMG_SIZE = 512                         # Rendered image resolution
NUM_IMAGES_PER_FILE = 40               # Per STL file
TRAIN_RATIO = 0.8                      # Train-validation split
SAVE_DIAGNOSTICS = True                # Saves visual diagnostics if True

## ⚙️ Requirements

Install dependencies using pip:

```bash
pip install trimesh pyrender opencv-python-headless albumentations matplotlib scikit-learn



              
