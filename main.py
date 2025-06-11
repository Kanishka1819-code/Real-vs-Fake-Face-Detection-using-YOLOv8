import math
import time
import cv2
import cvzone
from ultralytics import YOLO

# Load your trained model
model = YOLO('yolov8n.pt')  # Make sure the model is in the models folder

# Webcam config
cap = cv2.VideoCapture(0)  # Use 0 or 1 depending on your webcam
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Class labels
classNames = ["fake", "real"]
confidence = 0.6

# FPS calculation
prev_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        print("Failed to grab frame from camera")
        break

    # Make predictions
    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > confidence:
                label = classNames[cls]
                color = (0, 255, 0) if label == "real" else (0, 0, 255)

                # Draw box and label
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), colorC=color)
                cvzone.putTextRect(img, f'{label.upper()} {int(conf * 100)}%',
                                   (x1, max(35, y1)), scale=2, thickness=3,
                                   colorR=color, colorB=color)

    # FPS display
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the result
    cv2.imshow("Face Detection - Real vs Fake", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
