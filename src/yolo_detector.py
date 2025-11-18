import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_objects(frame):
    results = model(frame)
    return results

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_objects(frame)
        annotated = results[0].plot()

        cv2.imshow("YOLO Detection", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

