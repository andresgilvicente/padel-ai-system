from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

model.export(format='engine', device=0, half=True) 
