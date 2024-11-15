from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import cv2
# Load the model.
model = YOLO('yolov8n.pt') # transfer learning

# Training.
results = model.train(
    data='/PATH/TO/YOUR/DIRECTORY/data.yaml', # this is the path to your training data using roboflow to annotate 
    imgsz=640,                                # export it as YOLOv7 and you will get a data.yaml
    epochs=100,
    augment=True,
    batch=16,
    name='ChessCornerDetection',
    project='/PATH/TO/YOUR/DIRECTORY')
