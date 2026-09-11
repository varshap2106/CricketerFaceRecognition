"""
Flask Web Application & REST API for Indian Cricket Team Face Identification.
Provides image recognition endpoints, live webcam base64 feed processor, and player profile catalog.
"""

import os
import io
import base64
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BASE_DIR,
    WEB_APP_DIR,
    OUTPUTS_DIR,
    DEFAULT_PLAYER_PROFILES,
    PLAYER_DISPLAY_NAMES,
    MODEL_PATH,
    CLASS_DICT_PATH
)
from src.predict import FacePredictor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator

app = Flask(
    __name__,
    template_folder=str(WEB_APP_DIR / "templates"),
    static_folder=str(WEB_APP_DIR / "static")
)
CORS(app)

predictor = FacePredictor()


@app.route("/")
def index():
    """Serves the main application dashboard."""
    return render_template("index.html", players=DEFAULT_PLAYER_PROFILES)


@app.route("/api/players", methods=["GET"])
def get_players():
    """Returns metadata and statistics of all supported Indian cricketers."""
    return jsonify({
        "status": "success",
        "count": len(DEFAULT_PLAYER_PROFILES),
        "players": DEFAULT_PLAYER_PROFILES
    })


@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns current model status and dataset metrics."""
    model_ready = predictor.is_model_ready()
    classes = list(predictor.class_dict.keys()) if model_ready else []
    
    metrics = {}
    report_path = OUTPUTS_DIR / "evaluation_report.json"
    if report_path.exists():
        with open(str(report_path), "r") as f:
            metrics = json.load(f)

    return jsonify({
        "status": "success",
        "model_ready": model_ready,
        "classes_count": len(classes),
        "classes": classes,
        "metrics": metrics
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts image file or base64 data string, runs face detection & recognition,
    and returns annotated image + player identification data.
    """
    try:
        img_bgr = None

        # Check multipart file upload
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                in_memory_file = io.BytesIO()
                file.save(in_memory_file)
                data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
                img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # Check JSON base64 data (webcam snapshot)
        elif request.is_json and "image_base64" in request.json:
            b64_str = request.json["image_base64"]
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            data = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # Ensure predictor is up-to-date with latest trained model
        predictor._load_artifacts()

        # Run Prediction
        results = predictor.predict_image(img_bgr)
        
        # Convert annotated image to base64 for direct browser rendering
        annotated_bgr = results.get("annotated_image", img_bgr)
        _, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        response_detections = []
        for det in results.get("detections", []):
            x, y, w, h = det["box"]
            response_detections.append({
                "player_slug": det.get("player_slug", "unknown"),
                "player_name": det.get("player_name", "Unknown"),
                "confidence": det.get("confidence", 0.0),
                "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "profile": det.get("profile", {}),
                "top_predictions": det.get("top_predictions", [])
            })

        return jsonify({
            "status": "success",
            "faces_detected": results.get("faces_detected", 0),
            "detections": response_detections,
            "annotated_image_base64": f"data:image/jpeg;base64,{annotated_b64}"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/retrain", methods=["POST"])
def retrain():
    """Triggers dataset training and reloads model pipeline."""
    try:
        trainer = ModelTrainer()
        results = trainer.train_and_tune()
        
        # Reload predictor
        global predictor
        predictor = FacePredictor()
        
        # Run evaluator to update metrics
        evaluator = ModelEvaluator()
        evaluator.evaluate_model()

        return jsonify({
            "status": "success",
            "message": "Model successfully retrained",
            "benchmark": results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    """Serves generated confusion matrices and snapshots."""
    return send_from_directory(str(OUTPUTS_DIR), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Indian Cricket Face Recognition Web Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
