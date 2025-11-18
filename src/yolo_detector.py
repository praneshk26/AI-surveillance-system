import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

def detect_objects(frame):
    """
    Runs YOLO detection on a frame and extracts person detections.
    Returns:
        - person_boxes: list of (x1, y1, x2, y2)
        - cropped_persons: list of cropped frames for LSTM
        - annotated_frame: YOLO rendered frame
    """
    results = model(frame)[0]
    person_boxes = []
    cropped_persons = []

    for box in results.boxes:
        cls = int(box.cls)
        if cls == 0:  # class 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_boxes.append((x1, y1, x2, y2))

            crop = frame[y1:y2, x1:x2]
            cropped_persons.append(crop)

    annotated = results.plot()
    return person_boxes, cropped_persons, annotated

