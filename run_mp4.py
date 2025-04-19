from ultralytics import YOLO
import cv2

# === USER CONFIGURATION ===
model_path = "runs/detect/train10/weights/best.pt"  # path to your trained YOLO model
input_video_path = "sample_vid.mp4"              # path to input .mp4 file
output_video_path = "results_video.mp4"            # path to save the output

# === LOAD MODEL ===
model = YOLO(model_path)

# === LOAD VIDEO ===
cap = cv2.VideoCapture(input_video_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# === DEFINE VIDEO WRITER ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# === PROCESS FRAME-BY-FRAME ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model.predict(source=frame, conf=0.25, verbose=False)

    # Get the annotated frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

# === CLEANUP ===
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[INFO] Inference complete. Output saved to: {output_video_path}")

