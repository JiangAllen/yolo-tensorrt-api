from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import torch
import time
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

print("cuda initialization")
torch.cuda.init()
dummy = torch.randn(1).cuda()
torch.cuda.synchronize()
time.sleep(2)
print("initialization finished")

# model = YOLO("./yolov10m.pt").to(device) # 10~20, 96%~97%, yolov10n.pt
# model = YOLO("./yolov10m.pt")
# model.export(format="engine", half=True) # tensorrt
# model = YOLO("yolov10m.engine").to(device)
model = YOLO("./yolov10m.engine") # 30~40, 96%~97%

print("tensorrt loading")
dummy_image = torch.zeros((1, 3, 640, 640)).to(device)
model.predict(dummy_image)
torch.cuda.synchronize()
print("loading finished")

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

app = Flask(__name__)

@app.route("/predictimage", methods=["POST"])
def predictimage():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    file_bytes = file.read()

    image = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # result = model(image)
    result = model.predict(image, device=0)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    elapsed_time = max(end_time - start_time, 1e-3)
    fps = 1 / elapsed_time if elapsed_time > 0 else 0

    detections = []
    for r in result:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            class_id  = int(b.cls[0])
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "Unknown"
            detections.append({
                "classid": class_id,
                "classname": class_name,
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2]
            })

    return jsonify({
        "detections": detections,
        "fps": round(fps, 2),
        "device": device
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)