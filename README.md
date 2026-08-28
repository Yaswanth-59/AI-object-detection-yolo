# AI-Based Real-Time Object Detection System Using YOLO

A complete Flask + YOLO deep-learning project for detecting objects in uploaded images.

## Features

- YOLO deep-learning object detector
- Flask web application
- Upload an image
- Bounding boxes
- Object names
- Confidence percentages
- Clean responsive interface
- Optional dataset download for training practice

## 1. Create virtual environment

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

## 2. Install packages

```bash
pip install -r requirements.txt
```

## 3. Run

```bash
python app.py
```

Open:

http://127.0.0.1:5000

The first run downloads the YOLO11n pretrained weights automatically.

## 4. Test

Upload a photo containing objects such as a person, car, dog, bicycle, etc.

## Dataset note

A full COCO dataset is not included because it is very large. The pretrained YOLO model already contains learned weights for COCO's 80 classes. For training practice, run:

```bash
python download_dataset.py
```

See DATASET.md for details.

## Project title

AI-Based Real-Time Object Detection System Using YOLO Deep Learning

## Technologies

Python, Flask, YOLO, OpenCV, NumPy, Pillow, HTML, CSS
