from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import List
import os
import datetime
from ultralytics import YOLO

app = FastAPI()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Load the smallest YOLOv8 model

class ViolationDetectionModel:
    def detect(self, motorcycle_data):
        # This is where you'd implement your violation detection logic
        # For now, we'll return dummy data
        return [
            {"type": "triple_riding", "confidence": 0.8},
            {"type": "no_helmet", "confidence": 0.9}
        ]

violation_model = ViolationDetectionModel()

def detect_motorcycles(frame):
    results = model(frame, classes=[3])  # 3 is the class ID for motorcycles in COCO dataset
    motorcycles = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            motorcycles.append({
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "confidence": float(confidence)
            })
    return motorcycles

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    with open("temp_video.mp4", "wb") as buffer:
        buffer.write(await file.read())
    
    # Open the video file
    cap = cv2.VideoCapture("temp_video.mp4")
    
    violations = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect motorcycles
        detections = detect_motorcycles(frame)
        
        for detection in detections:
            # Check for violations
            detected_violations = violation_model.detect(detection)
            
            if detected_violations:
                # Save the frame as an image
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"violation_{timestamp}.jpg"
                cv2.imwrite(f"violations/{filename}", frame)
                
                violations.append({
                    "frame_time": cap.get(cv2.CAP_PROP_POS_MSEC),
                    "violations": detected_violations,
                    "image": filename
                })
    
    cap.release()
    os.remove("temp_video.mp4")
    
    return JSONResponse(content={"violations": violations})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)