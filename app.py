"""
Explainable and Environment-Aware Strawberry Disease Detection
Using an Enhanced ResNet9-SE Framework
==============================================================
Main Flask Application Server
"""

import os
import uuid
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

from model.predict import (
    analyze_image,
    get_model_status,
    get_preview_data,
    calculate_environmental_risk,
    DISEASE_CLASSES,
    CLASS_NAMES
)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()

# Upload Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# In-memory IoT Sensor Cache for ESP32 / DHT22 telemetry
LATEST_SENSOR_TELEMETRY = {
    "temperature": 23.4,
    "humidity": 82.5,
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "device_id": "ESP32_DHT22_FARM_NODE_01",
    "is_live_telemetry": False  # Flips to True when an ESP32 sends live packets
}


def allowed_file(filename: str) -> bool:
    """Verifies file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------------------------------------------------------------
# Frontend Page Routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    """Home Page with Agricultural AI theme, project overview, disease catalog."""
    return render_template(
        "index.html",
        active_page="home",
        diseases=DISEASE_CLASSES,
        model_status=get_model_status()
    )


@app.route("/detect")
def detect_page():
    """Disease Detection Page with drag-and-drop upload and live camera capture."""
    return render_template(
        "detect.html",
        active_page="detect",
        model_status=get_model_status()
    )


@app.route("/results")
def results_page():
    """
    Results Page layout prepared for future ResNet9-SE model & Grad-CAM integration.
    Supports preview mode to showcase the full UI layout and agronomic advisory.
    """
    disease_key = request.args.get("class", "Leaf Scorch")
    if disease_key not in DISEASE_CLASSES:
        disease_key = "Leaf Scorch"

    filename = request.args.get("file", "")
    image_url = f"/uploads/{filename}" if filename else None

    preview_info = get_preview_data(disease_key)
    model_status = get_model_status()

    return render_template(
        "results.html",
        active_page="results",
        disease_key=disease_key,
        data=preview_info,
        image_url=image_url,
        filename=filename,
        model_status=model_status,
        classes=CLASS_NAMES
    )


@app.route("/environment")
def environment_page():
    """Environmental Risk Assessment Page with microclimate disease risk gauges and ESP32 IoT info."""
    temp = float(request.args.get("temp", LATEST_SENSOR_TELEMETRY["temperature"]))
    hum = float(request.args.get("hum", LATEST_SENSOR_TELEMETRY["humidity"]))
    risk_data = calculate_environmental_risk(temp, hum)

    return render_template(
        "environment.html",
        active_page="environment",
        sensor_data=LATEST_SENSOR_TELEMETRY,
        risk_data=risk_data
    )


@app.route("/about")
def about_page():
    """About Project Page detailing ResNet9-SE, Grad-CAM, Environmental Awareness & SDG 2."""
    return render_template(
        "about.html",
        active_page="about",
        model_status=get_model_status(),
        classes=DISEASE_CLASSES
    )


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.route("/api/detect", methods=["POST"])
@app.route("/predict", methods=["POST"])
def api_detect():
    """
    Primary API endpoint for leaf image upload and detection analysis.
    Accepts:
      1. Multipart form file ('file' or 'image')
      2. JSON base64 encoded camera snapshot ('camera_image')
    """
    saved_filename = None
    file_path = None

    # Case 1: Standard File Upload via FormData
    if "file" in request.files or "image" in request.files:
        uploaded_file = request.files.get("file") or request.files.get("image")

        if uploaded_file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected. Please select a strawberry leaf photograph."
            }), 400

        if not allowed_file(uploaded_file.filename):
            return jsonify({
                "success": False,
                "error": f"Invalid file format '{uploaded_file.filename}'. Allowed formats: JPG, JPEG, PNG."
            }), 400

        # Generate unique secure filename
        ext = uploaded_file.filename.rsplit(".", 1)[1].lower()
        unique_id = uuid.uuid4().hex[:10]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"leaf_{timestamp}_{unique_id}.{ext}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
        uploaded_file.save(file_path)

    # Case 2: Camera Capture via Base64 Data URL
    elif request.is_json and request.json.get("camera_image"):
        data_url = request.json.get("camera_image")
        try:
            # Format: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
            header, encoded = data_url.split(",", 1)
            ext = "png"
            if "jpeg" in header or "jpg" in header:
                ext = "jpg"

            image_data = base64.b64decode(encoded)
            unique_id = uuid.uuid4().hex[:10]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_filename = f"cam_leaf_{timestamp}_{unique_id}.{ext}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)

            with open(file_path, "wb") as f:
                f.write(image_data)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Failed to process captured camera image: {str(e)}"
            }), 400

    else:
        return jsonify({
            "success": False,
            "error": "No image provided. Please upload an image or capture one using your camera."
        }), 400

    # Execute backend analysis pipeline
    analysis = analyze_image(file_path, saved_filename)
    return jsonify(analysis)


@app.route("/api/environment/risk", methods=["GET", "POST"])
def api_environment_risk():
    """Calculates disease risk from temperature and humidity inputs."""
    if request.is_json:
        data = request.get_json()
        temp = float(data.get("temperature", LATEST_SENSOR_TELEMETRY["temperature"]))
        hum = float(data.get("humidity", LATEST_SENSOR_TELEMETRY["humidity"]))
    else:
        temp = float(request.args.get("temperature", LATEST_SENSOR_TELEMETRY["temperature"]))
        hum = float(request.args.get("humidity", LATEST_SENSOR_TELEMETRY["humidity"]))

    result = calculate_environmental_risk(temp, hum)
    return jsonify({
        "success": True,
        "telemetry_source": "Live / Input query",
        "data": result
    })


@app.route("/api/environment/sensor-data", methods=["POST"])
def api_sensor_telemetry():
    """
    IoT Telemetry Ingestion endpoint for ESP32 + DHT22 weather station.
    Accepts JSON: {"temperature": 24.2, "humidity": 81.0, "device_id": "ESP32_FIELD_NODE"}
    """
    global LATEST_SENSOR_TELEMETRY
    data = request.get_json(silent=True) or {}

    if "temperature" not in data or "humidity" not in data:
        return jsonify({
            "success": False,
            "error": "Missing 'temperature' or 'humidity' values in telemetry payload."
        }), 400

    try:
        temp = float(data["temperature"])
        hum = float(data["humidity"])
        dev_id = str(data.get("device_id", "ESP32_DHT22_FARM_NODE"))

        LATEST_SENSOR_TELEMETRY = {
            "temperature": round(temp, 1),
            "humidity": round(hum, 1),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device_id": dev_id,
            "is_live_telemetry": True
        }

        risk = calculate_environmental_risk(temp, hum)

        return jsonify({
            "success": True,
            "message": "Telemetry received and environmental risk computed.",
            "sensor_telemetry": LATEST_SENSOR_TELEMETRY,
            "disease_risk": risk
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/model-status", methods=["GET"])
def api_model_status():
    """Returns the backend status of the ML model, PyTorch, and files."""
    return jsonify({
        "success": True,
        "status": get_model_status()
    })


@app.route("/uploads/<filename>")
def serve_upload(filename: str):
    """Securely serves uploaded leaf images."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# Error Handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": "File size exceeds the 16MB limit. Please upload a smaller photograph."
    }), 413


@app.errorhandler(404)
def not_found(error):
    return render_template("index.html", error_message="Page not found"), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("=" * 70)
    print("🍓 Strawberry Leaf Disease Detection Web Application")
    print("🔬 Enhanced ResNet9-SE Framework | Explainable AI & Environmental Awareness")
    print("=" * 70)
    print(f"📍 Local Server: http://127.0.0.1:{port}")
    print("📁 Uploads Directory:", UPLOAD_FOLDER)
    print("ℹ️  Model Integration Status:", get_model_status()["status_message"])
    print("=" * 70)
    app.run(host="0.0.0.0", port=port, debug=True)
