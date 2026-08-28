from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import os, uuid

app = Flask(__name__)
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_NAME = "yolo11n.pt"

print("Loading YOLO model...")
model = YOLO(MODEL_NAME)
print("Model loaded.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    if not file.filename:
        return "No image selected", 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        return "Unsupported image format", 400

    filename = f"{uuid.uuid4().hex}{ext}"
    input_path = os.path.join(UPLOAD_DIR, filename)
    file.save(input_path)

    results = model.predict(source=input_path, conf=0.25, save=True, project=UPLOAD_DIR, name="results", exist_ok=True)
    result = results[0]

    # YOLO saves the annotated image inside results/
    annotated_path = os.path.join(UPLOAD_DIR, "results", filename)
    if not os.path.exists(annotated_path):
        # fallback for formats where the saved name changes
        generated = result.save()
        annotated_path = generated if isinstance(generated, str) else input_path

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({
            "name": result.names[cls_id],
            "confidence": round(conf * 100, 2)
        })

    # Remove duplicate paths from URL handling
    annotated_rel = os.path.relpath(annotated_path, "static").replace("\\", "/")

    return render_template(
        "result.html",
        image_url="/static/" + annotated_rel,
        detections=detections,
        count=len(detections)
    )

@app.route("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
