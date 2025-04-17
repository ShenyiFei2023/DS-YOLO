from ultralytics import YOLO


model = YOLO('yolo11.yaml')
#model = YOLO('runs/train/haacs/weights/best.pt')
#model = YOLO('yolov8-aacs.yaml').load('runs/train/haacs/weights/best.pt')

results = model.train(data='dataset/SARSHIP-COCO/data.yaml',imgsz=640,epochs=300,patience=300,optimizer='SGD',save_json=True,save_txt=True)