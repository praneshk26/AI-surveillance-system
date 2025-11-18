import cv2
import torch
from yolo_detector import detect_objects
from feature_extractor import FeatureExtractor, preprocess_image
from lstm_model import ActivityLSTM
# from alert_system import play_alarm, send_email_alert  # Uncomment after UI team completes

# Load models
device = "cpu"
feature_model = FeatureExtractor().to(device)
lstm_model = ActivityLSTM().to(device)

SEQUENCE_LEN = 10
feature_sequence = []

def analyze_behavior(feature_vector):
    global feature_sequence

    feature_sequence.append(feature_vector)

    # Keep only last 10 frames
    if len(feature_sequence) > SEQUENCE_LEN:
        feature_sequence = feature_sequence[-SEQUENCE_LEN:]

    if len(feature_sequence) < SEQUENCE_LEN:
        return None  # not enough data yet

    seq_tensor = torch.stack(feature_sequence).unsqueeze(0)  # shape: (1,10,128)
    prediction = lstm_model(seq_tensor)
    
    prob_suspicious = prediction[0][1].item()

    return prob_suspicious


def start_surveillance():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not found!")
        return

    print("Surveillance running... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detections
        boxes, crops, annotated = detect_objects(frame)

        for crop in crops:
            img_tensor = preprocess_image(crop)
            if img_tensor is None:
                continue

            with torch.no_grad():
                features = feature_model(img_tensor)
                prob = analyze_behavior(features)

                if prob is not None:
                    if prob > 0.65:
                        print("⚠ SUSPICIOUS ACTIVITY DETECTED!")
                        # play_alarm()       # uncomment later
                        # send_email_alert() # uncomment later

        cv2.imshow("AI Surveillance", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_surveillance()
