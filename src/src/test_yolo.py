from yolo_detector import detect_objects
import cv2

cap = cv2.VideoCapture(0)
print("Testing YOLO... Press q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, crops, annotated = detect_objects(frame)
    cv2.imshow("YOLO Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
