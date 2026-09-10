from ultralytics import YOLO
import yaml

# Load the PyTorch model just to copy its internal dictionary of class names
model = YOLO("yolov8m.pt") 

metadata = {
    "description": "Ultralytics yolov8m model",
    "author": "Ultralytics",
    "task": "detect",
    "stride": 32,
    "batch": 1,
    "imgsz": [640, 640],
    "names": model.names
}

# Save it directly into the new folder
with open("yolov8m_hailo_model/metadata.yaml", "w") as f:
    yaml.dump(metadata, f)

print("Metadata created successfully!")
