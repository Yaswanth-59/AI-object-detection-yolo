"""
Downloads a small official Ultralytics detection dataset for testing/training practice.

IMPORTANT:
The YOLO11n model used by this project is already pretrained on the COCO
80-class dataset, so you do NOT need to download COCO just to run inference.

For a college training experiment, this script downloads COCO8 automatically.
For the complete COCO dataset, see DATASET.md.
"""
from ultralytics import settings
from ultralytics.data.utils import check_det_dataset

print("Downloading/checking COCO8 dataset...")
data = check_det_dataset("coco8.yaml")
print("Dataset ready:")
print(data)
