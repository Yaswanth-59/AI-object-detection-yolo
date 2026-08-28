# Dataset

## For immediate project demo
No dataset is required. `app.py` uses the pretrained YOLO11n model. The model supports the 80 object classes from COCO.

## For training practice
Run:

```bash
python download_dataset.py
```

This downloads the small COCO8 detection dataset automatically through Ultralytics.

## Full COCO dataset
The complete COCO detection dataset is very large and is intentionally NOT embedded in this ZIP. It contains tens of thousands of images and annotations and would make the project archive extremely large.

For a college project, use the pretrained YOLO model for the working demo, and use COCO8/COCO128 for training experiments. If your guide specifically requires training on the complete COCO dataset, download the official train/val images and annotations and configure the dataset YAML before training.
