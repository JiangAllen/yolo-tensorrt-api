# YOLO TensorRT Object Detection API

## 🧭 Overview
This project provides a **Flask-based REST API server** for performing real-time object detection using a **YOLOv10 model accelerated with TensorRT**. It also includes a **client script** that automatically sends local images to the server and displays detection results in JSON format. This setup is ideal for **offline GPU inference** with high performance (30–40 FPS on YOLOv10m engine).

---

## ✨ Features
- 🚀 High-speed inference using **TensorRT engine**
- 🎯 Supports standard **COCO classes (80 categories)**
- 🖼️ Accepts image uploads via REST API (`multipart/form-data`)
- 📊 Returns detection bounding boxes, class names, confidence scores, and FPS
- 💻 Client script for batch testing all images in a folder
- ⚙️ Auto GPU/CPU device selection (`cuda` fallback to `cpu`)

---

## 🏗️ Architecture

### 🖥️ `server.py`
- Initializes GPU and loads YOLOv10 TensorRT engine (`.engine` file)
- Provides an endpoint `/predictimage` for image detection
- Accepts image files, decodes them via OpenCV, and performs inference
- Returns detection results in structured JSON format with:
  ```json
  {
      "detections": [
          {"classid": 0, "classname": "person", "confidence": 0.97, "box": [x1, y1, x2, y2]}
      ],
      "fps": 33.3,
      "device": "cuda"
  }
  ```

### 💡 `client.py`
- Sends images to the Flask server via POST /predictimage
- Reads all .jpg, .jpeg, .png files in ./src/
- Prints formatted JSON detection results in the console

## ⚡ Quick Start
### 🧩 Prerequisites
- Python 3.9+
- GPU with CUDA support (recommended)
- Installed dependencies:
  ```
  pip install flask torch torchvision torchaudio ultralytics opencv-python requests
  ```

### ⚙️ Setup & Run
#### 1. Clone this repository
    ```
    git clone https://github.com/<your-username>/yolov10-tensorrt-api.git
    cd yolov10-tensorrt-api
    ```
#### 2. Prepare the YOLOv10 model
  - Place your trained TensorRT model file under the project directory:
```
./yolov10m.engine
```
 - You may also use .pt weights and export manually:
```
from ultralytics import YOLO
model = YOLO("yolov10m.pt")
model.export(format="engine", half=True)
```
#### 3. Start the server
  ```
  python server.py
  ```
  - Server will start at: http://0.0.0.0:8000
#### 4. Test with client
  - Place test images under ./src/
  - Run ```python client.py```
    
## 🧩 API Endpoints
### `POST /predictimage`
**Description:**  
Perform object detection on a single image.

**Request:**
```
Content-Type: multipart/form-data
file: <uploaded image>
```

**Response:**
```
{
  "detections": [
    {
      "classid": 0,
      "classname": "person",
      "confidence": 0.97,
      "box": [15, 20, 240, 400]
    }
  ],
  "fps": 35.2,
  "device": "cuda"
}
```

**Error Response:**
```{"error": "No file"}```

## 🛠️ Troubleshooting

| Issue | Possible Cause | Solution |
|-------|----------------|-----------|
| `RuntimeError: CUDA not available` | GPU driver or CUDA not installed | Check `torch.cuda.is_available()` |
| `Invalid image format` | Corrupted or unsupported image file | Ensure file is valid `.jpg` / `.png` |
| `No file` error | Missing `file` field in request | Use `multipart/form-data` upload |
| Slow FPS | CPU fallback or large image size | Ensure model runs on GPU / resize inputs |

## 📄 License
### This software is provided for research and demonstration purposes only.
