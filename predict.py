from ultralytics import YOLO

model = YOLO("runs/detect/seraphim_train/weights/best.pt")

#results = model.predict("datasets/drone-tracking/dataset1/cam0.mp4", save=True, show=True)
results = model.track("datasets/drone-tracking/dataset1/cam0.mp4", save=True, show=True, persist=True)
