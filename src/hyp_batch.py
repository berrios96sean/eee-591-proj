from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

# NOTE: Yolo models 4/7 use darknet and are not included with ultralytics.

# Load a COCO-pretrained YOLOv5n model
#model = YOLO("yolov3u.pt")
#model = YOLO("yolov5n.pt")
#model = YOLO("yolov6n.yaml")
#model = YOLO("yolov8n.pt")
#model = YOLO("yolov9c.pt")
#model = YOLO("yolov10n.pt")
#model = YOLO("yolov11n.pt")
#model = YOLO("yolo12n.pt")

# Display model information (optional)
#model.info()

# Train the model on the COCO8 example dataset for 100 epochs
#results = model.train(data="thermal_image_dataset.yaml", epochs=30, imgsz=640,batch=32)

# Validate the model
#metrics = model.val(data="thermal_image_dataset.yaml")
#print(metrics.box.map)  # map50-95i

# Run inference with the YOLOv5n model on the 'bus.jpg' image
#results = model("path/to/bus.jpg")

models = ["yolov9c.pt"]
batch_sizes = [8, 16, 32, 64]
map50s, precisions, recalls = [], [], []

for m in models:
    model = YOLO(m)
    map50s, precisions, recalls = [], [], []
    for b in batch_sizes:
        model.train(data="thermal_image_dataset.yaml", epochs=30, imgsz=640, batch=b, lr0=0.001, optimizer="Adam", workers=4,verbose=False)
        metrics = model.val(data="thermal_image_dataset.yaml")

        map50s.append(metrics.box.map50)
        precisions.append(metrics.box.mp)
        recalls.append(metrics.box.mr)

    # Create a dictionary with the metrics
    data = {
        "batch_size": batch_sizes,
        "mAP50": map50s,
        "precision": precisions,
        "recall": recalls
    }

    # Convert to a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv("hyp_batch_size_yolov9.csv", index=False)








