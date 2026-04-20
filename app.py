import os
# Fix for Render/Docker CPU thread freezing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import uuid
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO

app = Flask(__name__)

# Ensure upload directory exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the highly optimized ONNX YOLOv8 model to save memory
# This uses 4x less memory than PyTorch and avoids crashing the cloud server.
model = YOLO("yolov8n.onnx")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(request.url)
    
    file = request.files["image"]
    if file.filename == "":
        return redirect(request.url)
    
    if file:
        # Generate a unique filename to avoid overwriting
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Perform YOLOv8 Object Detection with reduced image size to save memory
        results = model(filepath, imgsz=320)
        
        # Save the result image with bounding boxes
        result_filename = "result_" + filename
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        results[0].save(filename=result_filepath)
        
        # Generate a summary of detected objects
        detected_objects = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            detected_objects.append({
                "name": class_name,
                "confidence": round(confidence * 100, 2)
            })
            
        return render_template("index.html", original_image=filename, result_image=result_filename, detected_objects=detected_objects)

if __name__ == "__main__":
    app.run(debug=True)
