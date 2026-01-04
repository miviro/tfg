import cv2
from ultralytics import YOLO

WEBCAM = 0

model = YOLO("runs/detect/seraphim_train/weights/best.pt")

cap = cv2.VideoCapture(WEBCAM)

while True:
    ret, frame = cap.read()
    # stream=True is efficient for video loops
    results = model(frame, stream=True)

    for r in results:
        annotated_frame = r.plot()
        cv2.imshow('yolocam', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
