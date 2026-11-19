# 🍓 Explainable and Environment-Aware Strawberry Disease Detection Using an Enhanced ResNet9-SE Framework

An academic mini-project web application built with **Python Flask**, **JavaScript (WebRTC & Canvas)**, and **PyTorch-ready modular architecture**. Designed for modern precision agriculture to identify foliar strawberry diseases, provide transparent visual explainability via **Grad-CAM**, and correlate microclimate risk factors.

---

## 🌟 Key Features

1. **Modern Agricultural AI Theme**:
   - Clean aesthetic with Forest Green, Sage Green, and Off-White palette.
   - Responsive layouts optimized for laptops, tablets, and smartphones in field conditions.
   - Comprehensive information on the 4 target strawberry foliar classes:
     - **Healthy Foliage** (*Fragaria × ananassa*)
     - **Leaf Scorch** (*Diplocarpon earlianum*)
     - **Powdery Mildew** (*Podosphaera aphanis*)
     - **Angular Leaf Spot** (*Xanthomonas fragariae*)

2. **Dual-Mode Foliage Capture**:
   - **Drag-and-Drop / File Picker**: Supports JPG, JPEG, and PNG files with frontend & backend file format and size validation (up to 16 MB).
   - **Live Camera Capture**: WebRTC `getUserMedia` with live framing target guide, front/rear camera switching, and instantaneous canvas snapshot acquisition.
   - **Instant Image Preview**: High-res preview with file metadata (dimensions, size, MIME type).
   - **Integration-Ready Analysis**: Displays a clean confirmation: *"Model integration pending. Your image has been uploaded successfully."* without faking predictions.

3. **Results & Explainable AI (Grad-CAM) Layout**:
   - Side-by-side comparison of input leaf vs. Grad-CAM saliency activation heatmap.
   - Disease diagnosis breakdown, classification confidence gauge, and full probability distribution.
   - Agronomic advisory with actionable next steps (bio-fungicide schedules, cultural practices, irrigation adjustments).
   - Interactive class switcher allowing evaluators to preview the UI layout across all 4 disease classes.

4. **Environmental Microclimate Risk Triangulation**:
   - Microclimate disease risk calculator based on temperature (°C) and relative humidity (RH %).
   - Computes risk levels (**Low**, **Medium**, **High**) using biological pathogen sporulation curves.
   - **IoT Ready**: Ingestion endpoint (`POST /api/environment/sensor-data`) ready for ESP32 microcontrollers with DHT22 sensors.

5. **Research & SDG 2 Alignment**:
   - Detailed architecture review of the **Enhanced ResNet9-SE** with Squeeze-and-Excitation channel attention.
   - Comparative benchmark framework against MobileNetV3-Small and EfficientNet-B0.
   - Alignment with **UN Sustainable Development Goal 2: Zero Hunger** (reducing crop loss and chemical runoff).

---

## 📂 Project Directory Structure

```
berry/
│
├── app.py                      # Primary Flask application server & API routes
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation & setup instructions
│
├── model/                      # Modular machine learning architecture
│   ├── __init__.py
│   ├── resnet9_se.py           # PyTorch ResNet9 architecture with Squeeze-and-Excitation (SE) block
│   ├── predict.py              # Preprocessing pipeline, model loader, Grad-CAM & risk assessment
│   └── weights/                # Directory where 'resnet9_se.pth' will be placed
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Shared layout, navbar, mobile drawer & footer
│   ├── index.html              # Home page with hero, workflow pipeline & disease catalog
│   ├── detect.html             # Disease detection page (Drag-drop upload & live WebRTC camera)
│   ├── results.html            # Diagnostic report & Grad-CAM heatmap visualization
│   ├── environment.html        # Microclimate disease risk dashboard & IoT ESP32 guide
│   └── about.html              # Academic paper breakdown, ResNet9-SE theory & SDG 2
│
├── static/                     # Static assets
│   ├── css/
│   │   └── style.css           # Premium agricultural design stylesheet
│   └── js/
│       ├── main.js             # Navigation & global utilities
│       ├── detect.js           # File upload, validation, preview & analysis submission
│       └── camera.js           # WebRTC live camera capture & snapshot logic
│
└── uploads/                    # Secure local storage for uploaded leaf photographs
```

---

## 🚀 Getting Started Locally

### 1. Prerequisites
- Python 3.9+ installed on your system.

### 2. Setup Virtual Environment

```bash
# Navigate to the project directory
cd /path/to/berry

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows (PowerShell):
# .\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask Web Application

```bash
python3 app.py
```

### 5. Open in Your Browser
Visit [http://127.0.0.1:5001](http://127.0.0.1:5001) in your web browser (default port 5001 avoids macOS AirPlay Receiver port 5000 conflicts).

---

## 🔬 Future Model Integration: Step-by-Step Guide

The backend is modular and decoupled from the web layer. When your PyTorch training is complete, connect your model in three simple steps:

### Step 1: Install PyTorch and Computer Vision Libraries
```bash
pip install torch torchvision opencv-python
```

### Step 2: Save Your Model Checkpoint
When training your `ResNet9SE` model in PyTorch, save the state dictionary:

```python
import torch
from model.resnet9_se import ResNet9SE

# After training loop:
torch.save(model.state_dict(), "model/weights/resnet9_se.pth")
```

### Step 3: Automatic Activation
Once `model/weights/resnet9_se.pth` is present:
- `model/predict.py` automatically detects the weights file.
- The `/api/detect` endpoint automatically runs real-time inference on uploaded leaf images.
- Grad-CAM hooks into the final convolutional layer (`Stage 3 Residual Conv2d`) to output actual heatmap overlays.
- **No frontend rewrites or code changes are required!**

---

## 📡 IoT Microclimate Telemetry (ESP32 + DHT22)

To send real-time field weather readings into the Environmental Risk dashboard:

**API Endpoint**: `POST /api/environment/sensor-data`  
**Content-Type**: `application/json`  
**Payload Example**:
```json
{
  "temperature": 23.4,
  "humidity": 82.5,
  "device_id": "ESP32_FIELD_NODE_01"
}
```

---

## 🎓 Academic Credit & SDG Contribution

- **Project**: Explainable and Environment-Aware Strawberry Disease Detection Using an Enhanced ResNet9-SE Framework
- **Domain**: Precision Agriculture, Deep Learning, Edge Computing & Explainable AI (XAI)
- **Global Goal**: UN Sustainable Development Goal 2: Zero Hunger (Target 2.4)
