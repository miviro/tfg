from ultralytics import YOLO

model = YOLO("runs/detect/train8/weights/best.pt")

results = model("/home/miviro/Desktop/tfg/yolo/datasets/Anti-UAV-RGBT/val/20190926_195921_1_5/visible.mp4", save=True, show=True)
